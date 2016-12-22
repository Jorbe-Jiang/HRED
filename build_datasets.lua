-------------------------------------------
--generate formated datasets for torch7
-------------------------------------------

function split_line(line)
	local values = {}
	values = string.split(line, '([%s]+)')
	for i = 1, #values do
		values[i] = tonumber(values[i])
	end
	return values
end

function padding_zero(Uxs, Uxs_maxlen, left_or_right)
	Uxs_len = table.getn(Uxs)
	for i = 1, Uxs_len do
		for j = table.getn(Uxs[i])+1, Uxs_maxlen do
			if left_or_right == 'right' then
				Uxs[i][j] = 0
			else
				table.insert(Uxs[i], 1, 0)
			end
		end
	end

	return Uxs
end

function deepcopy(object)
 	local lookup_table = {}

 	local function _copy(object)
	 	if type(object) ~= "table" then
	 	    return object
	 	elseif lookup_table[object] then
		    return lookup_table[object]
		end
		local new_table = {}
		lookup_table[object] = new_table
		for index, value in pairs(object) do
		    new_table[_copy(index)] = _copy(value)
		end
		return setmetatable(new_table, getmetatable(object))
	end

    return _copy(object)
end

function fsize(file)
	local size = 0
	for line in io.lines(file) do
		size = size + 1
	end
	return size
end

function gen_batches_data(train_valid_test)
	U1s_batches_enc = {}
	U2s_batches_enc = {}
	U2s_batches_dec = {}
	U3s_batches_dec = {}
	U2s_batches_tar = {}
	U3s_batches_tar = {}
	nbatches = 0

	local open_file
	local output_file
	if train_valid_test == 'train' then
		open_file = opt.train_corpus
		output_file = './datas/'..tostring(opt.batch_size)..'_batches_train.t7'
	elseif train_valid_test == 'valid' then
		open_file = opt.valid_corpus
		output_file = './datas/'..tostring(opt.batch_size)..'_batches_valid.t7'
	else
		open_file = opt.test_corpus
		output_file = './datas/'..tostring(opt.batch_size)..'_batches_test.t7'
	end

	local U1s_enc = {}
	local U2s_enc = {}
	local U2s_dec = {}
	local U3s_dec = {}
	local U2s_tar = {}
	local U3s_tar = {}
	local U1s_maxlen = 0
	local U2s_maxlen = 0
	local U3s_maxlen = 0 
	local i = 1 --line index
	
	-- get 1/4 of the all datasets for test
	require 'math'
	local all_lines = fsize(open_file)
	local stop_line = math.floor(all_lines/4)

	for line in io.lines(open_file) do
		local words_idx = split_line(line)
		words_len = table.getn(words_idx)
		if i % 3 == 1 then
			table.remove(words_idx)
			table.insert(U1s_enc, words_idx)
			if words_len > U1s_maxlen then
				U1s_maxlen = words_len
			end 
		elseif i % 3 == 2 then
			local words_idx_clone_1 = deepcopy(words_idx)
			local words_idx_clone_2 = deepcopy(words_idx)
			table.remove(words_idx_clone_1)
			table.insert(U2s_enc, words_idx_clone_1)
			table.insert(U2s_tar, words_idx_clone_2)
			table.insert(words_idx, 1, 1)
			table.remove(words_idx)
			table.insert(U2s_dec, words_idx)
			if words_len > U2s_maxlen then
				U2s_maxlen = words_len
			end 
		else
			local words_idx_clone = deepcopy(words_idx)
			table.insert(U3s_tar, words_idx_clone)
			table.insert(words_idx, 1, 1)
			table.remove(words_idx)
			table.insert(U3s_dec, words_idx)
			if words_len > U3s_maxlen then
				U3s_maxlen = words_len
			end 
		end
			
		if i % (opt.batch_size*3) == 0 then
			table.insert(U1s_batches_enc, torch.Tensor(padding_zero(deepcopy(U1s_enc), U1s_maxlen, 'left')):t())
			table.insert(U2s_batches_enc, torch.Tensor(padding_zero(deepcopy(U2s_enc), U2s_maxlen, 'left')):t())
			table.insert(U2s_batches_dec, torch.Tensor(padding_zero(deepcopy(U2s_dec), U2s_maxlen, 'right')):t())
			table.insert(U3s_batches_dec, torch.Tensor(padding_zero(deepcopy(U3s_dec), U3s_maxlen, 'right')):t())
			table.insert(U2s_batches_tar, torch.Tensor(padding_zero(deepcopy(U2s_tar), U2s_maxlen, 'right')):t())
			table.insert(U3s_batches_tar, torch.Tensor(padding_zero(deepcopy(U3s_tar), U3s_maxlen, 'right')):t())
			U1s_enc = {}
			U2s_enc = {}
			U2s_tar = {}
			U3s_tar = {}
			U2s_dec = {}
			U3s_dec = {}
			U1s_maxlen = 0
			U2s_maxlen = 0
			U3s_maxlen = 0
			nbatches = nbatches + 1
		end
		i = i + 1
		-- for test
		if i == stop_line then
			break
		end
	end
	--[[
	print(U1s_batches_enc[1])
	print(U1s_batches_enc[2])
	print(U2s_batches_enc[1])
	print(U2s_batches_enc[2])
	print(U2s_batches_dec[1])
	print(U2s_batches_dec[2])
	print(U3s_batches_dec[1])
	print(U3s_batches_dec[2])
	print(U2s_batches_tar[1])
	print(U2s_batches_tar[2])
	print(U3s_batches_tar[1])
	print(U3s_batches_tar[2])
	os.exit() 
	]]--
	
	U1s_enc = nil
	U2s_enc = nil
	U2s_tar = nil
	U3s_tar = nil
	U2s_dec = nil
	U3s_dec = nil
	
	torch.save(output_file, {U1s_batches_enc, U2s_batches_enc, U2s_batches_dec, U3s_batches_dec, U2s_batches_tar, U3s_batches_tar, nbatches})

end