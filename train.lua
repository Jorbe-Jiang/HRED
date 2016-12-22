require 'torch';
require 'rnn';
require 'dpnn'
require 'optim'
require 'cutorch' 
require 'string';



cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a HRED model')
cmd:text()
cmd:text('Options: ')

-- model params
cmd:option('-enc_hidden_size', 256, 'size of encoder LSTM internal state') --if load model, then must be same with loaded model
cmd:option('-dec_hidden_size', 256, 'size of decoder LSTM internal state') --if load model, then must be same with loaded model
cmd:option('-context_hidden_size', 256, 'size of context LSTM internal state(same with dec_hidden_size)') --if load model, then must be same with loaded model
cmd:option('-word_dim', 100, 'word vector dim')
cmd:option('-vocab_size', 5003, 'vocabulary size(5000 + start/end/unknow signal)')
cmd:option('-batch_size', 16, 'number of sequences to train on in parallel')
cmd:option('-num_epochs', 200, 'number of epochs through the training data')
cmd:option('-load_model', 0, 'whether to load trained model to init the training model')
cmd:option('-load_model_file', './model/tmp_1.model', 'choose load_model_file to init the training model')
cmd:option('-load_embeddings', 1, 'whether to load pre-trained word vector')
cmd:option('-load_embeddings_file', './datas/word_vec.npy', 'choose load_embedding_file to init the embeddings')
cmd:option('-train_corpus', './datas/train.t7', 'train corpus')
cmd:option('-valid_corpus', './datas/valid.t7', 'valid corpus')
cmd:option('-test_corpus', './datas/test.t7', 'test corpus')
cmd:option('-print_every', 50, 'print the predict results every 5 epochs')
cmd:option('-eval_every', 20, 'evaluate the model every 10 epochs')
cmd:option('-train_or_test', 'train', 'training or testing')


-- optimization
cmd:option('-lr', 0.1, 'learning rate')
cmd:option('-lr_decay_rate', 0.99, 'Decay learning rate')
cmd:option('-start_decay', 0, 'Start decay the learning rate')
cmd:option('-start_decay_at', 20, 'Start decay after this epoch')
cmd:option('-grad_clip', 2, 'clip gradients at this value, pass 0 to disable')
cmd:option('-drop_rate', 0.2, 'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-param_init', 0.1, 'Parameters are initialized over uniform distribution with support (-param_init, param_init)')
cmd:option('-seed', 16, 'torch manual random number generator seed')
cmd:option('-cell', 'LSTM', 'LSTM or GRU or FastLSTM') --if load model, then must be same with loaded model
cmd:option('-optimizer', 'rmsprop', 'optimizer function')

-- GPU/CPU
cmd:option('-gpu_id', 0, 'which gpu to use. -1 = use CPU') 

--others
cmd:option('-version', 'HRED model: use wikipedia word vector to init and train the word vector with model', 'depict of this version demo')

cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
print(opt)

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpu_id >= 0 then
	local ok, cunn = pcall(require, 'cunn')
	local ok2, cutorch = pcall(require, 'cutorch')
	if not ok then print('package cunn not found!') end
	if not ok2 then print('package cutorch not fount!') end
	if ok and ok2 then
		print('Using CUDA on GPU '..opt.gpu_id..'...')
		cutorch.setDevice(opt.gpu_id + 1) --GPU id 1-indexed
		cutorch.manualSeed(opt.seed)
	else
		print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        	print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        	print('Falling back on CPU mode')
        	opt.gpu_id = -1 -- overwrite user setting
	end
end



function train_model()
	--[[
	require 'build_datasets'
	print('Gen batches data...')
	gen_batches_data('train')
	gen_batches_data('valid')
	gen_batches_data('test')
	print('Gen batches data successfully...')
	os.exit()
	]]--

	npy4th = require 'npy4th';
	local funcs = loadfile('build_model.lua')
	funcs()
	local model, criterion = build()

	require 'encoder-decoder'
	
	print('Begin to load training datasets and valid datasets...')
	local batches_train_data = torch.load('./datas/'..tostring(opt.batch_size)..'_batches_train.t7')
	--local batches_train_data = torch.load('./datas/'..tostring(opt.batch_size)..'_batches_test.t7') --for test
	local batches_valid_data = torch.load('./datas/'..tostring(opt.batch_size)..'_batches_valid.t7')
	print('Load datasets successfully ...')
	
	train(model, criterion, batches_train_data, batches_valid_data)
end


function test_model()
	require 'eval'

	print('Begin to load testing ndatasets ...')
	local batches_test_data = torch.load('./datas/'..tostring(opt.batch_size)..'_batches_test.t7')
	print('Load datasets successfully ...')

	print('Load trained model...')
	assert(path.exists(opt.load_model_file), 'check the model file path')
	print('Loading model from: '..opt.load_model_file..'...')
	local model_and_opts = torch.load(opt.load_model_file)
	local model = model_and_opts[1]
	
	test(model, batches_test_data)
end

-------------------------------------------------------------
if opt.train_or_test == 'train' then
	train_model()
else
	test_model()
end
	
	
