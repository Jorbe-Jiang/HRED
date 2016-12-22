--------------------------------------------------
--training process
--------------------------------------------------
--------------------------------------------------
--some extra functions
--------------------------------------------------
function idx_2_word()
	local vocab_file = './datas/VOCAB.dat'
	local idx_2_word = {}
	for line in io.lines(vocab_file) do
		table.insert(idx_2_word, line)
	end
	return idx_2_word
end

function word_2_idx()
	local vocab_file = './datas/VOCAB.dat'
	local word_2_idx = {}
	local word_idx = 1
	for line in io.lines(vocab_file) do
		word_2_idx[line] = word_idx
		word_idx = word_idx + 1
	end
	return word_2_idx
end

function forward_connect(hred_enc_rnn, dec_rnns, i)
	dec_rnns[i].userPrevOutput = nn.rnn.recursiveCopy(dec_rnns[i].userPrevOutput, hred_enc_rnn.outputs[i])
	if opt.cell ~= 'GRU' then
		dec_rnns[i].userPrevCell = nn.rnn.recursiveCopy(dec_rnns[i].userPrevCell, hred_enc_rnn.cells[i])
	end
end

function backward_connect(hred_enc_rnn, dec_rnns, i)
	if opt.cell ~= 'GRU' then
		hred_enc_rnn.userNextGradCell = nn.rnn.recursiveCopy(hred_enc_rnn.userNextGradCell, dec_rnns[i].userGradPrevCell)
	end
	hred_enc_rnn.gradPrevOutput = nn.rnn.recursiveCopy(hred_enc_rnn.gradPrevOutput, dec_rnns[i].userGradPrevOutput)
end

local idx_2_word_table = idx_2_word()
-- U3s_words_idx:[n_U3s, seqLen]
function translate_to_word(U3s_words_idx)
	local U3s_words_table = {}
	local n_U3s = U3s_words_idx:size(1)
	local seq_len = U3s_words_idx:size(2)
	for i = 1, n_U3s do
		local U3_words = {}
		for j = 1, seq_len do
			if U3s_words_idx[{i, j}] ~= 1 then
				local word = idx_2_word_table[U3s_words_idx[{i, j}]]
				table.insert(U3_words, word)
			end
		end
		table.insert(U3s_words_table, U3_words)
	end
	return U3s_words_table
end

--training 
function train(model, criterion, batches_train_data, batches_valid_data)
	print('Begin to train...')
	
	--hred_enc_inputs:<U1, U2> {Tensor(U1_seqLen, batchSize), Tensor(U2_seqLen, batchSize)}
	--dec_inputs:<U2, U3> {Tensor(U2_seqLen, batchSize), Tensor(U3_seqLen, batchSize)}
	--dec_tar_outputs:<U2, U3> {Tensor(U2_seqLen, batchSize), Tensor(U3_seqLen, batchSize)}
	local hred_enc_inputs, dec_inputs, dec_tar_outputs
	local hred_enc_outputs 
	local dec_outputs = {} -- one batch data decoder outputs probilities {Tensor(U2_seqLen, batchSize, vocabSize), Tensor(U3_seqLen, batchSize, vocabSize)}

	--model components
	local hred_enc, hred_enc_rnn, dec, dec_rnns, params, grad_params
	
	hred_enc = model[1]
	hred_enc_rnn = model[2]
	dec = model[3]
	dec_rnns = model[4]
	params = model[5]
	grad_params = model[6]

	--update decoder 1
	function feval_1(x)
		assert(x == params)
		if x == nil then
			print('Notes: params is nil!!!')
			os.exit()
		end
		
		--reset params gradients
		grad_params:zero()
	
		--forward pass and backward pass
		hred_enc_outputs = hred_enc:forward(hred_enc_inputs)
		forward_connect(hred_enc_rnn, dec_rnns, 1)
		dec_outputs[1] = dec:get(1):get(1):forward(dec_inputs[1])
		
		if opt.gpu_id >= 0 then
			dec_outputs[1] = dec_outputs[1]:cuda()
		end
		collectgarbage()
		
		local train_loss_1 = criterion:forward(dec_outputs[1], dec_tar_outputs[1])
		local grad_output_1 = criterion:backward(dec_outputs[1], dec_tar_outputs[1])

        --print(torch.max(grad_output_1))
        --print(torch.min(grad_output_1))
		if opt.grad_clip > 0 then grad_output_1:clamp(-opt.grad_clip, opt.grad_clip) end

		dec:get(1):get(1):backward(dec_inputs[1], grad_output_1)
		backward_connect(hred_enc_rnn, dec_rnns, 1)
		local zero_tensor = opt.gpu_id >= 0 and torch.CudaTensor(hred_enc_outputs):zero() or torch.Tensor(hred_enc_outputs):zero()
		hred_enc:backward(hred_enc_inputs, zero_tensor) 

		--cilp gradient 
		if opt.grad_clip > 0 then grad_params:clamp(-opt.grad_clip, opt.grad_clip) end

		return train_loss_1, grad_params
	end
	
	--update decoder 2
	function feval_2(x)
		assert(x == params)
		if x == nil then
			print('Notes: params is nil!!!')
			os.exit()
		end
		
		--reset params gradients
		grad_params:zero()

		--forward pass and backward pass
		hred_enc_outputs = hred_enc:forward(hred_enc_inputs)
		forward_connect(hred_enc_rnn, dec_rnns, 2)	
		dec_outputs[2] = dec:get(1):get(2):forward(dec_inputs[2])

		if opt.gpu_id >= 0 then
			dec_outputs[2] = dec_outputs[2]:cuda()
		end
		collectgarbage()
		
		local train_loss_2 = criterion:forward(dec_outputs[2], dec_tar_outputs[2])
		local grad_output_2 = criterion:backward(dec_outputs[2], dec_tar_outputs[2])

        --print(torch.max(grad_output_2))
        --print(torch.min(grad_output_2))
		if opt.grad_clip > 0 then grad_output_2:clamp(-opt.grad_clip, opt.grad_clip) end

		dec:get(1):get(2):backward(dec_inputs[2], grad_output_2)
		backward_connect(hred_enc_rnn, dec_rnns, 2)
		local zero_tensor = opt.gpu_id >= 0 and torch.CudaTensor(hred_enc_outputs):zero() or torch.Tensor(hred_enc_outputs):zero()
		hred_enc:backward(hred_enc_inputs, zero_tensor)
		
		--cilp gradient
		if opt.grad_clip > 0 then grad_params:clamp(-opt.grad_clip, opt.grad_clip) end

		return train_loss_2, grad_params
	end

	--evaluate
	function eval_loss(x)
		assert(x == params)
		if x == nil then
			print('Notes: params is nil!!!')
			os.exit()
		end
			
		--forward pass
		hred_enc_outputs = hred_enc:forward(hred_enc_inputs)
		forward_connect(hred_enc_rnn, dec_rnns, 1)
		forward_connect(hred_enc_rnn, dec_rnns, 2)
		dec_outputs[1] = dec:get(1):get(1):forward(dec_inputs[1])
		dec_outputs[2] = dec:get(1):get(2):forward(dec_inputs[2])
		if opt.gpu_id >= 0 then
			dec_outputs[1] = dec_outputs[1]:cuda()
			dec_outputs[2] = dec_outputs[2]:cuda()
		end
		collectgarbage()
		
		local train_loss_1 = criterion:forward(dec_outputs[1], dec_tar_outputs[1])
		local train_loss_2 = criterion:forward(dec_outputs[2], dec_tar_outputs[2])
		local norm_loss_1 = train_loss_1*opt.batch_size/torch.sum(torch.gt(dec_inputs[1],0))
		local norm_loss_2 = train_loss_2*opt.batch_size/torch.sum(torch.gt(dec_inputs[2],0))
		local norm_loss = norm_loss_1 + norm_loss_2
		return norm_loss
	end

	function argmax(pred_outs)
		max_values, max_idxs = torch.max(pred_outs, 3)
		return max_idxs
	end

	function print_pred_results()
		U3s_pred_pros = dec_outputs[2]
		U3s_pred_idx = argmax(U3s_pred_pros)
		U3s_pred_idx = U3s_pred_idx:view(U3s_pred_idx:size(1), opt.batch_size):t()
		if opt.gpu_id >= 0 then
			U3s_pred_idx = U3s_pred_idx:cudaLong()
			U3s_true_idx = dec_tar_outputs[2]:t():cudaLong()
		else
			U3s_pred_idx = U3s_pred_idx:long()
			U3s_true_idx = dec_tar_outputs[2]:t():long()		
		end

		local print_U3s_pred_words_table = translate_to_word(U3s_pred_idx:narrow(1, 1, 3))
		local print_U3s_true_words_table = translate_to_word(U3s_true_idx:narrow(1, 1, 3))
		print('U3s pred(3 arrow):')
		--print(U3s_pred_idx:narrow(1, 1, 3))
		for i = 1, #print_U3s_pred_words_table do
			print(i..'> '..table.concat(print_U3s_pred_words_table[i], ' '))
		end
		print('U3s true(3 arrow):')
		--print(U3s_true_idx:narrow(1, 1, 3))
		for i = 1, #print_U3s_true_words_table do
			print(i..'. '..table.concat(print_U3s_true_words_table[i], ' '))
		end
		print(U3s_pred_idx:ne(U3s_true_idx):sum())
	end

	--training datasets 
	local U1s_batches_enc, U2s_batches_enc, U2s_batches_dec, U3s_batches_dec, U2s_batches_tar, U3s_batches_tar, nbatches
	U1s_batches_enc = batches_train_data[1]
	U2s_batches_enc = batches_train_data[2]
	U2s_batches_dec = batches_train_data[3]
	U3s_batches_dec = batches_train_data[4]
	U2s_batches_tar = batches_train_data[5]
	U3s_batches_tar = batches_train_data[6]
	nbatches = batches_train_data[7]
	
	--evaluate datasets
	local valid_U1s_batches_enc, valid_U2s_batches_enc, valid_U2s_batches_dec, valid_U3s_batches_dec, valid_U2s_batches_tar, valid_U3s_batches_tar, valid_nbatches
	valid_U1s_batches_enc = batches_valid_data[1]
	valid_U2s_batches_enc = batches_valid_data[2]
	valid_U2s_batches_dec = batches_valid_data[3]
	valid_U3s_batches_dec = batches_valid_data[4]
	valid_U2s_batches_tar = batches_valid_data[5]
	valid_U3s_batches_tar = batches_valid_data[6]
	valid_nbatches = batches_valid_data[7]


	local date = os.date("%m_%d") --today's date
	local train_losses = {}
	local eval_losses = {}
	local prev_eval_loss, curr_eval_loss
	local optim_state = {learningRate = opt.lr}
	local start_time = os.time() --begin time

	for i = 1, opt.num_epochs do
		hred_enc:training()
		dec:training()
		local batches_order = torch.randperm(nbatches) --shuffle the order of all batches data
		
		if (i >= opt.start_decay_at or opt.start_decay == 1) and opt.lr > 0.001 then
			opt.lr = opt.lr * opt.lr_decay_rate
			optim_state.learningRate = opt.lr
		end
		
		local epoch_losses = {}
		
		for j = 1, nbatches do
			hred_enc_inputs = {U1s_batches_enc[batches_order[j]], U2s_batches_enc[batches_order[j]]}
			dec_inputs = {U2s_batches_dec[batches_order[j]], U3s_batches_dec[batches_order[j]]}
			dec_tar_outputs = {U2s_batches_tar[batches_order[j]], U3s_batches_tar[batches_order[j]]}
			
			if opt.gpu_id >= 0 then
				for i = 1, 2 do
					hred_enc_inputs[i] = hred_enc_inputs[i]:int():cuda()
					dec_inputs[i] = dec_inputs[i]:int():cuda()
					dec_tar_outputs[i] = dec_tar_outputs[i]:int():cuda()
				end
			end
            
            --[[
            print'1111111111111111'
            print(torch.sum(params))
            print(torch.max(params))
            print(torch.min(params))
            print(torch.max(grad_params))
            print(torch.min(grad_params))
            print'111111111111111111111'
            ]]--

			--update params of decoder 1
			local _1, loss_1 = optim[opt.optimizer](feval_1, params, optim_state)
            
            --[[
            print'222222222222222222222222'
            print(torch.sum(params))
            print(torch.max(params))
            print(torch.min(params))
            print(torch.max(grad_params))
            print(torch.min(grad_params))
            print'22222222222222222'
            ]]--
            
            collectgarbage()
			--update params of decoder 2
			local _2, loss_2 = optim[opt.optimizer](feval_2, params, optim_state)
            
            --[[
            print'3333333333333333333333'
            print(torch.sum(params))
            print(torch.max(params))
            print(torch.min(params))
            print(torch.max(grad_params))
            print(torch.min(grad_params))
            print'33333333333333333333333'
            ]]--

            collectgarbage()
			local norm_loss_1 = loss_1[1]*opt.batch_size/torch.sum(torch.gt(dec_inputs[1],0))
			local norm_loss_2 = loss_2[1]*opt.batch_size/torch.sum(torch.gt(dec_inputs[2],0))
			local norm_train_loss = norm_loss_1 + norm_loss_2
			
			if norm_train_loss < -0.2 or i % opt.print_every == 0 then
				print_pred_results()
			end

			epoch_losses[j] = norm_train_loss
			
			local msg = string.format("Epoch: %d, complete: %.3f%%, lr = %.4f, train_loss = %6.4f, grad norm = %6.2e ", i, 100*j/nbatches, opt.lr, norm_train_loss, torch.norm(model[6]))
			io.stdout:write(msg..'\b\r')
			io.flush()
		end

		table.insert(train_losses, torch.mean(torch.Tensor(epoch_losses)))

		if i % opt.eval_every == 0 or i == opt.num_epochs then
			print('Evaluating ...')
			
			for k = 1, valid_nbatches do
				hred_enc:evaluate()
				dec:evaluate()
				hred_enc_inputs = {valid_U1s_batches_enc[k], valid_U2s_batches_enc[k]}
				dec_inputs = {valid_U2s_batches_dec[k], valid_U3s_batches_dec[k]}
				dec_tar_outputs = {valid_U2s_batches_tar[k], valid_U3s_batches_tar[k]}
				local norm_e_loss = eval_loss(params)
				collectgarbage()
				print("eval loss: ", norm_e_loss)
				table.insert(eval_losses, norm_e_loss)
			end
			
			local mean_loss = torch.mean(torch.Tensor(eval_losses))
			
			if prev_eval_loss == nil then
				prev_eval_loss = mean_loss
				curr_eval_loss = mean_loss
			else
				curr_eval_loss = mean_loss
			end

			--early stoping
			if prev_eval_loss > 0 then
                if curr_eval_loss > prev_eval_loss * 3 then
                    print('Loss is exploding, early stoping...')
                    os.exit()
                end
            end

            if prev_eval_loss < 0 then
                if curr_eval_loss > prev_eval_loss/3 then
                    print('Loss is exploding, early stoping...')
                    os.exit()
                end
            end

			--start decay learning rate
			if curr_eval_loss > prev_eval_loss then
				opt.start_decay = 1
			else
				opt.start_decay = 0 
			end
		
			prev_eval_loss = curr_eval_loss			

			local save_file = string.format("./model/%s_epoch_%d_epochs_%d_%.2f_model.t7", date, i, opt.num_epochs, mean_loss)
			print('Saving model to: ', save_file)
			hred_enc:clearState()
			hred_enc_rnn:clearState()
			dec:clearState()
			dec_rnns[1]:clearState()
			dec_rnns[2]:clearState()
			dec_rnns[1]:double()
			dec_rnns[2]:double()
			local package_model = {
				hred_enc:double(),        --hred_enc
				dec:double(),             --dec
				hred_enc_rnn:double(),    --hred_enc_rnn
				dec_rnns                  --dec_rnns
			}
			torch.save(save_file, {package_model, opt})
			local tmp_result_file = string.format("./results/%s_epoch_%d_epochs_%d_%d_tmp_train_result.npy", date, i,  opt.num_epochs, opt.batch_size)
			print('Saving tmp results to: ', tmp_result_file)
			npy4th.savenpy(tmp_result_file, torch.Tensor(train_losses))
			print('Evaluating end ...')
			eval_losses = {}
		end
		collectgarbage()
	end

	--saving results
	local result_file = string.format("./results/%s_%d_%d_train_result.npy", date, opt.num_epochs, opt.batch_size)
	print('Saving results to: ', result_file)
	npy4th.savenpy(result_file, torch.Tensor(train_losses))
	local end_time = os.time()
	local train_time = os.difftime(end_time, start_time)
	print("Training cost :", string.format("%.2d:%.2d:%.2d", train_time/(60*60), train_time/60%60, train_time%60))
	print('Training end ...')
end	
