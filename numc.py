import numpy as np
from scipy import weave
import sys

#===============================================================================
# Hack: Everything which we do not implement on our own gets forward to numpy.
# TODO: also enable "from numpyng import *" 
class ModuleWrapper:
	def __init__(self, inner_module):
		self.inner_module = inner_module

	def __getattr__(self, name):
		try:
			return getattr(self.inner_module, name)
		except AttributeError:
			return getattr(np, name)

sys.modules[__name__] = ModuleWrapper(sys.modules[__name__])


#===============================================================================
class ArrayExpression:
	""" The Base-Class """
	def __add__(self, other):
		return(add(self, other))
	#TODO: also implement all other special functions like __sub__, __not__,...
	
	
	def materialize(self):
		""" Should evaluate self for all indices. Results are provided via __array_interface__ """  
		pass #TODO


#===============================================================================
class CodeBuilder():
	""" Centerpiece during generation of C-Code """
	def __init__(self):
		self.code = ""
		self.args = {}
		self.uid_counter = 0

	def uid(self):
		""" generate a new, unique identifier """
		self.uid_counter += 1
		return("uid%d"%self.uid_counter)
		
	def add_arg(self, arg):
		""" Registers arg, which interfaces with python-code """ 
		uid = self.uid()
		self.args[uid] = arg
		return(uid)
			
	def write(self, code):
		self.code += code

	def writeln(self, code):
		self.write(code+"\n")
	
	def build(self, array, index):
		""" If array is a numpy.ndarray, its code gets build here. """
		if(isinstance(array, ArrayExpression)):
			uid = array.build(self, index)
			
		elif(isinstance(array, np.ndarray)):
			#TODO: find a better place for this code
			#TODO: support non continuous arrays
			#TODO: support any thing that porvides __array_interface__
			arg_uid = self.add_arg(array)
			uid = self.uid()
			index_code = index[0]
			#index_code = "0"
			for (s,i) in zip(array.shape, index)[1:]:
				index_code = "( %s ) * %s + %s"%(index_code, s, i)
			self.writeln("%s %s = %s[%s];"%(array.dtype, uid, arg_uid, index_code))
		else:
			raise(Exception("Unkown type: %s"%array))
		return(uid)
		
	def run(self):
		self.code = self.code.replace("float64", "double") #TODO: solve genericly
		print "Running:\n"+ self.code
		#print self.args
		weave.inline(self.code, self.args.keys(), self.args,
				force=False, verbose=2)
					#type_converters=weave.converters.blitz, compiler = 'gcc', verbose=2)


#===============================================================================
class UnaryOperation(ArrayExpression):
	def __init__(self, arg):
		self.arg = arg
		self.shape = arg.shape
		self.dtype = arg.dtype
	
	def build(self, builder, index):
		arg_uid = builder.build(self.arg, index)
		code = self.__class__.template % (arg_uid)
		uid = builder.uid()
		builder.writeln("%s %s = %s;"%(self.dtype, uid, code))
		return(uid)
		
#===============================================================================
class BinaryOperation(ArrayExpression):
	def __init__(self, arg1, arg2):
		(self.arg1, self.arg2) = (arg1, arg2)
		if(arg1.dtype != arg2.dtype): raise(Exception("Casting is not supported, yet"))
		self.dtype = arg1.dtype
		self.broadcast = Broadcast(self.arg1.shape, self.arg2.shape)
		self.shape = self.broadcast.shape
	
	def build(self, builder, index):
		index1 = self.broadcast.index1(index)
		index2 = self.broadcast.index2(index)
		arg1_uid = builder.build(self.arg1, index1)
		arg2_uid = builder.build(self.arg2, index2)
		code = self.__class__.template % (arg1_uid, arg2_uid)
		uid = builder.uid()
		builder.writeln("%s %s = %s;"%(self.dtype, uid, code))
		return(uid)

#===============================================================================
class Broadcast:
	""" Takes care of NumPy-broadcasting """
	def __init__(self, shape1, shape2):
		s1 = list(shape1)
		s2 = list(shape2)
		
		if(len(s1) < len(s2)):
			s1 = [1]*(len(s2) - len(s1)) + s1
		if(len(s2) < len(s1)):
			s2 = [1]*(len(s1) - len(s2)) + s2
		
		shape = []
		self.broadcasted = []
		for (i,j) in zip(s1,s2):
			if(i==j):
				shape.append(i)
				self.broadcasted.append(0) # not broadcasted
			elif(i==1):
				shape.append(j)
				self.broadcasted.append(1) #arg1 broadcasted
			elif(j==1):
				shape.append(i)
				self.broadcasted.append(2) #arg2 broadcasted
			else:
				raise(Exception("Could not broadcast"))
		
		self.shape = tuple(shape)

	def index1(self, index):
		new_index = []
		for (i, b) in zip(index, self.broadcasted):
			if(b != 1):
				new_index.append(i)
		return(tuple(new_index))
	
	def index2(self, index):
		new_index = []
		for (i, b) in zip(index, self.broadcasted):
			if(b != 2):
				new_index.append(i)
		return(tuple(new_index))
	
#===============================================================================
#TODO write more like these
class add(BinaryOperation):
	template = "( %s + %s )"

class sin(UnaryOperation):
	template = "sin(%s)"

class square(UnaryOperation):
	template = "pow(%s, 2)"


#===============================================================================
def sum(a, axis=None, dtype=None, out=None):
	#TODO support axis != None
	if(isinstance(a, np.ndarray)):
		print("sum: Is a numpy array - forwarding")
		return(np.sum(a, axis, dtype, out))
	if(not dtype): dtype = a.dtype
	out_shape = (1)
	if(not out): out = np.empty(out_shape, dtype)
	out[0] = 0.0
	B = CodeBuilder()
	b = B.add_arg(out) # output-array
	index = []
	for N in a.shape:
		n = B.add_arg(N) #length of loop
		i = B.uid()  #loop-variable
		index.append(i)
		B.writeln("for (int %s=0; %s<%s; %s++)  {"%(i,i,n,i))

	a_uid = B.build(a, index)
	B.writeln("%s[0] += %s ;"%(b, a_uid))
	
	#debuging	
	#for (k,i) in enumerate(index):
	#	B.writeln("std::cout << %s << \": \" << %s << std::endl;"%(k,i))
	#B.writeln("std::cout <<  %s[0] << std::endl;"%b)
	#B.writeln("std::cout <<  std::endl;")
	
	for N in a.shape:
		B.writeln("}")
	B.run()  #compile and run code
	return(out)

#EOF