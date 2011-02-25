import numpy as np
from scipy import weave
import sys

#===============================================================================
# Hack: Everything which we do not implement on our own gets forward to numpy.
# TODO: also enable "from numc import *" 
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
	
	@property
	def __array_interface__(self):
		if(not hasattr(self, "result")):
			self.evaluate()
		return(self.result.__array_interface__)
		 
	def evaluate(self):
		""" Evaluate itself for all indices. Results are accessable via __array_interface__ """
		#print("evaluating")
		out = np.empty(self.shape, self.dtype)
		builder = CodeBuilder()
		index = builder.loop(self)
		builder.writeln("{")
		a_uid = self.build(builder, index)
		ArrayWrapper(out).build_inline(builder, index)
		builder.writeln("= %s;"%a_uid)
		builder.writeln("}")
		builder.run()  #compile and run code
		self.result = out
	

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

	def loop(self, arg):
		index = []
		for N in arg.shape:
			n = self.add_arg(N) #length of loop
			i = self.uid()  #loop-variable
			index.append(i)
			self.writeln("for (int %s=0; %s<%s; %s++) "%(i,i,n,i))
		return(index)
	
	def run(self):
		self.code = self.code.replace("float64", "double") #TODO: solve genericly
		#print "Running:\n"+ self.code
		#print self.args
		weave.inline(self.code, self.args.keys(), self.args,
				force=False, verbose=2)
					#type_converters=weave.converters.blitz, compiler = 'gcc', verbose=2)

#===============================================================================
class ArrayWrapper:
	""" Wrapper to handle e.g. numpy.ndarray objects transparently. """
	#TODO: support any thing that porvides __array_interface__
	def __init__(self, array):
		self.array = array
		self.shape = array.shape
		self.dtype = array.dtype
	
	def build_inline(self, builder, index):
		arg_uid = builder.add_arg(self.array)
		index_code = index[0]
		for (s,i) in zip(self.shape, index)[1:]:
			index_code = "( %s ) * %s + %s"%(index_code, s, i)
		builder.write("%s[%s]"%(arg_uid, index_code))
	
	def build(self, builder, index):
		uid = builder.uid()
		builder.write("%s %s = "%(self.dtype, uid))
		self.build_inline(builder, index)
		builder.writeln(";")
		return(uid)
	

#===============================================================================
class UnaryOperation(ArrayExpression):
	def __init__(self, arg):
		if(not isinstance(arg, ArrayExpression)):  arg = ArrayWrapper(arg)
		self.arg = arg
		self.shape = arg.shape
		self.dtype = arg.dtype
		
		
	def build(self, builder, index):
		arg_uid = self.arg.build(builder, index)
		code = self.__class__.template % (arg_uid)
		uid = builder.uid()
		builder.writeln("%s %s = %s;"%(self.dtype, uid, code))
		return(uid)
		
#===============================================================================
class BinaryOperation(ArrayExpression):
	def __init__(self, arg1, arg2):
		if(not isinstance(arg1, ArrayExpression)):  arg1 = ArrayWrapper(arg1)		
		if(not isinstance(arg2, ArrayExpression)):  arg2 = ArrayWrapper(arg2)
		(self.arg1, self.arg2) = (arg1, arg2)
		if(arg1.dtype != arg2.dtype): raise(Exception("Casting is not supported, yet"))
		self.dtype = arg1.dtype
		self.broadcast = Broadcast(self.arg1.shape, self.arg2.shape)
		self.shape = self.broadcast.shape
	
	def build(self, builder, index):
		index1 = self.broadcast.index1(index)
		index2 = self.broadcast.index2(index)
		arg1_uid = self.arg1.build(builder, index1)
		arg2_uid = self.arg2.build(builder, index2)
		code = self.__class__.template % (arg1_uid, arg2_uid)
		uid = builder.uid()
		builder.writeln("%s %s = %s;"%(self.dtype, uid, code))
		return(uid)

#===============================================================================
class Broadcast:
	""" Takes care of NumPy-broadcasting """
	def __init__(self, shape1, shape2):
		self.fill = len(shape1) - len(shape2)
		s1 = [1] * max(0, -1*self.fill) + list(shape1)
		s2 = [1] * max(0, self.fill) + list(shape2)
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
		new_index = list( index )
		for (i, b) in enumerate(self.broadcasted):
			if(b == 1): new_index[i]="0"
		fill = abs(min(self.fill, 0))
		return( tuple(new_index[fill:]) )
	
	def index2(self, index):	
		new_index = list( index )
		for (i, b) in enumerate(self.broadcasted):
			if(b == 2): new_index[i]="0"
		fill = max(self.fill, 0)
		return( tuple(new_index[fill:]) )
	
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
	if(not isinstance(a, ArrayExpression)):
		print("sum: Not an ArrayExpression - forwarding to NumPy")
		return(np.sum(a, axis, dtype, out))
	if(not dtype): dtype = a.dtype
	out_shape = (1)
	if(not out): out = np.empty(out_shape, dtype)
	out[0] = 0.0
	B = CodeBuilder()
	b = B.add_arg(out) # output-array
	index = B.loop(a)
	B.writeln("{")
	a_uid = a.build(B, index)
	B.writeln("%s[0] += %s;"%(b, a_uid))
	#debuging	
	#for (k,i) in enumerate(index):
	#	B.writeln("std::cout << %s << \": \" << %s << std::endl;"%(k,i))
	#B.writeln("std::cout <<  %s[0] << std::endl;"%b)
	#B.writeln("std::cout <<  std::endl;")
	B.writeln("}")
	B.run()  #compile and run code
	return(out)

#EOF