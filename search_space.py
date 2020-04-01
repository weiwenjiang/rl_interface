import copy

def combination(length):#寻找1-9的排列组合，length代表想剪枝的weight数
	data = [i for i in range(1,10)]#代表3*3kernel的位置
	result = []
	temp = [0] * length
	l = len(data)
	def next_num(li = 0,ni = 0):
		if ni == length:
			result.append(copy.copy(temp))
			return
		for lj in range(li, l):
			temp[ni] = data[lj]
			next_num(lj+1,ni+1)
	next_num()
	return result

def search_space():#搜索剪枝空间
	space = {}#以剪枝位置和kernel形式存储
	result = []#所有可能的剪枝情况，[0]表示不做剪枝
	for i in range(10):#每种排列组合均加入搜索空间，即任意剪枝位置
		combi_list = combination(i)
		for e in combi_list:
			result.append(e)
	for e in result:
		key = ''#作为搜索空间的key
		temp = [[1 for i in range(3)]for j in range(3)]#3*3的列表，每个位置代表剪枝或保留
		if not e:
			key = '0'
		else:
			for e1 in e:
				key = key + str(e1)
				if e1 == 1:temp[0][0] = 0
				if e1 == 2:temp[0][1] = 0
				if e1 == 3:temp[0][2] = 0
				if e1 == 4:temp[1][0] = 0
				if e1 == 5:temp[1][1] = 0
				if e1 == 6:temp[1][2] = 0
				if e1 == 7:temp[2][0] = 0
				if e1 == 8:temp[2][1] = 0
				if e1 == 9:temp[2][2] = 0
		space[key] = temp
	return space

def get_pattern(str):#可以输入0或1-9的组合，例如：‘0’表示不剪枝，‘12345’表示剪枝12345位置的weight，顺序需要从大到小
	space = search_space()
	# print(space)
	return space[str]

# print(get_pattern('4567'))




	