----------------------------------------------------
--model structure
----------------------------------------------------
local emb = emb or nn.LookupTableMaskZero(opt.vocab_size, opt.word_dim):setMaxNorm(2)
if opt.load_embeddings == 1 then
	print('Loading pre-trained word vectors from: '..opt.load_embeddings_file)
	local trained_word_vecs = npy4th.loadnpy(opt.load_embeddings_file)
	emb.weight = trained_word_vecs
end

function RNN_elem(recurrence)
	local utterance_rnn = nn.Sequential()
	local hred_enc_embeddings = nn.LookupTableMaskZero(opt.vocab_size, opt.word_dim):setMaxNorm(2)
	hred_enc_embeddings.weight = emb.weight:clone()
	utterance_rnn:add(hred_enc_embeddings)

    --batch norm
    --utterance_rnn:add(nn.Sequencer(nn.BatchNormalization(opt.word_dim)))

	local rnn = recurrence(opt.word_dim, opt.enc_hidden_size)
	utterance_rnn:add(nn.Sequencer(rnn:maskZero(1)))

    --batch norm
    --utterance_rnn:add(nn.Sequencer(nn.BatchNormalization(opt.enc_hidden_size)))

	if opt.drop_rate > 0 then
		utterance_rnn:add(nn.Sequencer(nn.Dropout(opt.drop_rate)))
	end

	utterance_rnn:add(nn.Select(1, -1))

	return utterance_rnn
end

--build hred encoder(utterance encoder and context encoder)
function build_hred_encoder(recurrence)
	local hred_enc = nn.Sequential()
	local hred_enc_rnn
	local par = nn.ParallelTable()
	
	--build parallel utterance rnns
	local rnns = {}
	for i = 1, 2 do
		table.insert(rnns, RNN_elem(recurrence))
	end

	rnns[2] = rnns[1]:clone('weight', 'bias', 'gradWeight', 'gradBias') --utterance 2 rnn share the weight with utterance 1 rnn

	for i = 1, 2 do
		par:add(rnns[i])
	end

	hred_enc:add(par)
	
	hred_enc:add(nn.JoinTable(1, 2))
	hred_enc:add(nn.Reshape(2, opt.batch_size, opt.enc_hidden_size))

	--build context layer
	local context_layer = nn.Sequential()
	hred_enc_rnn = recurrence(opt.enc_hidden_size, opt.context_hidden_size)
	context_layer:add(nn.Sequencer(hred_enc_rnn:maskZero(1)))

    --batch norm
    --context_layer:add(nn.Sequencer(nn.BatchNormalization(opt.context_hidden_size)))
    
	if opt.drop_rate > 0 then
		context_layer:add(nn.Sequencer(nn.Dropout(opt.drop_rate)))
	end
	hred_enc:add(context_layer)

	return hred_enc, hred_enc_rnn
	
end


--build decoders(utterance 1 decoder and utterance 2 decoder)
function build_decoder(recurrence)
	local dec = nn.Sequential()
	local dec_rnns = {}  --decoder 1 rnn and decoder 2 rnn
	local dec_embeddings = nn.LookupTableMaskZero(opt.vocab_size, opt.word_dim):setMaxNorm(2)
	dec_embeddings.weight = emb.weight:clone()
	
	local par = nn.ParallelTable()

	--build parallel decoders
	for i = 1, 2 do
		local dec_rnn = nn.Sequential()
		dec_rnn:add(dec_embeddings)	
         --batch norm
        --dec_rnn:add(nn.Sequencer(nn.BatchNormalization(opt.word_dim)))

		local rnn = recurrence(opt.word_dim, opt.dec_hidden_size)
		table.insert(dec_rnns, rnn)
		dec_rnn:add(nn.Sequencer(rnn:maskZero(1)))
         --batch norm
        --dec_rnn:add(nn.Sequencer(nn.BatchNormalization(opt.dec_hidden_size)))
		if opt.drop_rate > 0 then
			dec_rnn:add(nn.Sequencer(nn.Dropout(opt.drop_rate)))
		end
		local linear = nn.Linear(opt.dec_hidden_size, opt.vocab_size)
		dec_rnn:add(nn.Sequencer(nn.MaskZero(linear, 1)))
		dec_rnn:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(), 1)))
		par:add(dec_rnn)
	end

	dec:add(par)
	
	return dec, dec_rnns
end


--build model 
function build()
	local recurrence = nn[opt.cell]
	print('Building model...')
	print('RNN type: '..opt.cell)
	print('Vocab size: '..opt.vocab_size)
	print('Embedding size: '..opt.word_dim)
	print('Encoder layer hidden size: '..opt.enc_hidden_size)
	print('Context layer hidden size: '..opt.context_hidden_size)
	print('Decoder layer hidden size: '..opt.dec_hidden_size)
	
	--criterion
	local criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1))
	
	local hred_enc, hred_enc_rnn, dec
	local dec_rnns = {}
	
	-- whether to load pre-trained model from load_model_file
	if opt.load_model == 0 then
		hred_enc, hred_enc_rnn = build_hred_encoder(recurrence)
		dec, dec_rnns = build_decoder(recurrence)
		
	else
		--load the trained model
		assert(path.exists(opt.load_model_file), 'check the model file path')
		print('Loading model from: '..opt.load_model_file..'...')
		local model_and_opts = torch.load(opt.load_model_file)
		local model, model_opt = model_and_opts[1], model_and_opts[2]
		
		
		--load the model components
		hred_enc = model[1]:double()
		dec = model[2]:double()
		hred_enc_rnn = model[3]:double()
		dec_rnns[1] = model[4][1]:double()
		dec_rnns[2] = model[4][2]:double()
		--batch_size may be changed
		hred_enc:remove(3) 
		hred_enc:insert(nn.Reshape(2, opt.batch_size, opt.enc_hidden_size), 3)
		
	end
	
	local layers = {hred_enc, dec}
	
	--run on GPU
	if opt.gpu_id >= 0 then
		for i = 1, #layers do
			layers[i]:cuda()
		end
		criterion:cuda()
	end
	local Model = nn.Sequential()
	Model:add(hred_enc)
	Model:add(dec)
	local params, grad_params = Model:getParameters()

	
	if opt.gpu_id >= 0 then
		params:cuda()
		grad_params:cuda()
	end

	--package model for training
	local model = {
		hred_enc,
		hred_enc_rnn,
		dec,
		dec_rnns,
		params,
		grad_params
	}
	
	print('Building model successfully...')
	return model, criterion
end
