# -*- coding: utf-8 -*-

from scipy import weave
import sys
import itertools
import numpy
import traceback
from time import time #TODO remove
import __builtin__

#import math
#===============================================================================
# Idea: generate and compile also similar code-snippets for future use

#===============================================================================
# Hack: Everything which we do not implement on our own gets forward to numpy.
# TODO: also enable "from numc import *"
# 
# def apply2any(something, func):
	# """ applies func to something or if its iterable to all its elements """
	# if(not hasattr(something, '__iter__')):
		# something[i] = func(x)
	# else:
		# for (i,x) in enumerate(something):
			# somethingt[i] = func(x)
	# return(something)
# 
#===============================================================================

DEBUG = 0

def set_debug(d):
	global DEBUG
	DEBUG = d

class ModuleWrapper:
	def __init__(self, inner_module):
		self.inner_module = inner_module

	def __getattr__(self, name):
		try: #TODO: evtl mit hasattr
			return getattr(self.inner_module, name)
		except AttributeError:
			return AssimilateDecorator( getattr(numpy, name) )
		


class AssimilateDecorator:
	def __init__(self, inner_func):
		self.inner_func = inner_func

	def __call__(self, *args, **kwargs):
		result = self.inner_func(*args, **kwargs)
		#TODO result might be iterable
		if(isinstance(result, numpy.ndarray)):
			result = NumpyArray(result)
			#result = ndarray(NumpyArray(result))
		return result


#TODO: for testing: lets do everything on our own
sys.modules[__name__] = ModuleWrapper(sys.modules[__name__])


newaxis = numpy.newaxis

#===============================================================================
def assimilate(something):
	""" converts something somehow to an ArrayExpression """
	assert(not isinstance(something, ArraySource))
	
	if(isinstance(something, ndarray)):
		return(something)
	
	#if(isinstance(something, ArrayExpression)):
	#	return(something)
	if(not isinstance(something, numpy.ndarray)):
		something = numpy.array(something)
	return( NumpyArray(something) )
		#return( ndarray(NumpyArray(something)) )

#===============================================================================
class ArraySource(object):
	""" The Base-Class """
	def __new__(cls, *args, **kwargs):
		#print("__new__ called: "+cls.__name__)
		new_obj = object.__new__(cls)
		cls.__init__(new_obj, *args, **kwargs)
		return ndarray(new_obj)
	
	def build_shape(self, builder):
		return [builder.add_arg(n) for n in self.shape]
	 	
		# assert(False)
		# foo = [builder.add_arg(n) for n in self.shape]
		# foo2 = []
		# for x in foo:
			# foo2.append( x +"c" )
			# builder.writeln("const int %sc = %s;"%(x, x))
			# 
		# return(foo2)
		# #return [builder.add_arg(n) for n in self.shape]
#===============================================================================
class ArrayExpression(ArraySource):
	def evaluate(self):
		""" Evaluate itself for all indices. Results are accessable via __array_interface__ """
		if(DEBUG): print("evaluating: %s"%str(self))
		out = empty(self.shape, self.dtype)
		B = CodeBuilder()
		index = B.loop(self)
		B.writeln("{")
		a_uid = self.build_get(B, index)
		out.src.build_set(B, index, a_uid)
		B.writeln("}")
		B.run()  #compile and run code
		return( out )
		
#===============================================================================
""" Now we can exchange Expression for NumpyArrays after evaluation """
class ndarray(object):
	def __init__(self, source):
		assert(isinstance(source, ArraySource))
		self.src = source
	
	def copy(self):
		# the actual copy is done later, either by __array_interface__ or __setitem__
		return( ndarray(self.src) )
	
	@property
	def shape(self): return self.src.shape
	
	@property
	def dtype(self): return self.src.dtype
	
	@property
	def ndim(self):	
		return( len(self.shape) )
	
	@property
	def size(self):
		return( int(numpy.prod(self.shape)) )
	
	
	def __add__(self, other): return( add(self, other) )
	def __sub__(self, other): return( sub(self, other) )
	def __div__(self, other): return( div(self, other) )
	def __mul__(self, other): return( mul(self, other) )
	
	#def flaten(self): return( ravel(self.copy()) )
	@property
	def flat(self): return( ravel(self) )
		
	
	def __getitem__(self, slices):
		if(DEBUG):	print "__getitem__(%s) called"%str(slices)
		return( Slicing(self, slices) )
		
		
	def __setitem__(self, key, value):
		if(DEBUG): print "__setitem__(%s, %s) called"%(key, value)
		value = assimilate(value)
		if(not isinstance(self.src, NumpyArray)):
			self.src = self.src.evaluate().src
			
		#now self.src is a NumpyArray for sure !
		
		if(sys.getrefcount(self.src) > 2): #one for self.src and one for getrefcount()
			#print "Makeing a copy!!!!!!!!!"
			t1 = time()
			self.src = NumpyArray(self.src.array.copy()).src
			t2 = time()
			print "Made a Copy it took: "+str(t2-t1)
		
		#print("running setitem")
		out = Slicing(self, key)
		B = CodeBuilder()
		index = B.loop(out)
		B.writeln("{")
		a_uid = value.src.build_get(B, index)
		out.src.build_set(B, index, a_uid)
		B.writeln("}")
		B.run()  #compile and run code
		
	
	def __str__(self):
	 	return(str(self.src))


	def get_array(self):
		try:
			if(not isinstance(self.src, NumpyArray)):
				self.src = self.src.evaluate().src
		except:
			print("!!!!!!!!!! An exception in mk_array occured !!!!!!!!!!!!!")
			traceback.print_exc()
		return(self.src.array)
	
	@property
	def __array_interface__(self): return self.get_array().__array_interface__ 
	
	
	def __lt__(self, other): raise(NotImplementedError)	
	def __le__(self, other): raise(NotImplementedError)
	def __eq__(self, other): raise(NotImplementedError)
	def __ne__(self, other): raise(NotImplementedError)
	def __gt__(self, other): return self.get_array().__gt__(other)
	def __ge__(self, other): raise(NotImplementedError)
#===============================================================================
class NumpyArray(ArraySource):
	""" Wrapper to handle e.g. numpy.ndarray objects transparently. """
	#TODO: support any thing that provides __array_interface__
	# maybe we can use numpy.array( ) here as well
	def __init__(self, array):
		assert(isinstance(array, numpy.ndarray))
		self.array = array
		self.shape = array.shape
		self.dtype = array.dtype 
	
	def build_get(self, builder, index):
		tmp_uid = builder.uid("tmp")
		elem = self.mk_element(builder, index)
		builder.writeln("%s %s = %s;"%(self.dtype, tmp_uid, elem))
		return(tmp_uid)
	
	def build_set(self, builder, index, value):
		elem = self.mk_element(builder, index)
		builder.writeln("%s = %s;"%(elem, value))
		
	def mk_element(self, builder, index):
		arg_uid = builder.add_arg(self.array)
		if(len(index) == 0):
			return("*"+arg_uid)
		index_list = ["S%s[%d]*(%s)"%( arg_uid, n, i) for (n,i) in enumerate(index)]
		index_code = " + ".join(index_list) 	
		#weave.inline casted arg allready to dtype but strides are in byte
		#return("%s[(%s)/sizeof(%s)]"%(arg_uid, index_code, self.dtype))
		return("*((%s*)(((char*)%s)+(%s)))"%(self.dtype, arg_uid, index_code))

	def build_shape(self, builder): #TODO: improve
		self_uid = builder.add_arg(self.array)
		return ["N%s[%d]"%(self_uid,i) for i in range(len(self.shape))]
	
	def __str__(self):
		if(len(self.shape) == 0):
			return( str(self.array) )
		#return(str(self.array))
		return("NumpyArray(shape=%s)"%str(self.array.shape))

#===============================================================================
class CodeBuilder():
	""" Centerpiece during generation of C-Code """
	def __init__(self):
		self.code = ""
		self.args = {}
		self.uids = set()
	
	#def insert(self, arg, index):
	#	#TODO: check cache - if elements is allready there - maybe we need contexts 
	#	return arg.build(self, index)
		
	def uid(self, name="uid"):
		""" generate a new, unique identifier """
		found_name = name
		for i in itertools.count(2):
			if(found_name not in self.uids):
				self.uids.add(found_name)
				return(found_name)
			found_name = name+str(i)
		
	def add_arg(self, arg, name="arg"):
		""" Registers arg, which interfaces with python-code """
		if(isinstance(arg, numpy.ndarray)):
			for (k,v) in self.args.items():
				if(id(v) == id(arg)): # was arg already added earlier?
					return(k) 
		uid = self.uid(name)
		self.args[uid] = arg
		return(uid)
			
	def write(self, code):
		self.code += code

	def writeln(self, code):
		self.write(code+"\n")

	def loop(self, arg):
		index = []
		#for n in arg.build_shape(self):
		for N in arg.shape:
			if(N == 1):
				index.append(0)
			else:
				n = self.add_arg(int(N), "N") #length of loop
				i = self.uid("i")  #loop-variable
				index.append(i)
				self.writeln("for (int %s=0; %s<%s; %s++) "%(i,i,n,i))
			
		return(tuple(index))
	
	def run(self):
		self.code = self.code.replace("float", "float64") #TODO: solve genericly
		self.code = self.code.replace("float64", "double") #TODO: solve genericly
		self.code = self.code.replace("double64", "double") #TODO: solve genericly
		self.code = self.code.replace("int32", "int") #TODO: solve genericly
		
		if(DEBUG):
			print ("START"+"="*60)
			
			for (k,v) in self.args.items():
				if(isinstance(v, int)):
					print "%s ->  %s"%(k,v)
				if(isinstance(v, numpy.ndarray)):
					for i in range(v.ndim):
						self.writeln('std::cout << "S'+k+"[%d] = "%i+'"<< '+"S"+k+"[%d]"%i+"<<std::endl;")
					if(v.ndim == 0):
						print "%s ->  %s"%(k,v)
					else:
						print "%s ->  %s"%(k,v.shape)
			print "Running:\n"+ self.code	
			#print self.args.keys()
		t1 = time()
		weave.inline(self.code, self.args.keys(), self.args, verbose=DEBUG,)
			#extra_compile_args=["-O3"], )
				#force=False, verbose=DEBUG) 
					#type_converters=weave.converters.blitz, compiler = 'gcc', verbose=2)

		t2 = time()
		print "C-Run took: "+str(t2-t1)
		if(DEBUG):
			print ("END"+"="*60)
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
class UnaryOperation(ArrayExpression):
	def __init__(self, ufunc, arg):
		self.arg = assimilate(arg).copy()
		self.shape = self.arg.shape
		self.ufunc = ufunc
		self.refcount = 0
		if(isinstance(self.ufunc.outtype, numpy.dtype)):
			self.dtype = self.ufunc.outtype 
		else:
			self.dtype = self.arg.dtype
		
	def build_get(self, builder, index):
		arg_uid = self.arg.src.build_get(builder, index)
		code = self.ufunc.template % {"arg":arg_uid}
		uid = builder.uid("tmp")
		builder.writeln("%s %s = %s;"%(self.dtype, uid, code))
		return(uid)
		 
	def build_shape(self, builder):
		return(self.arg.src.build_shape(builder))

	def __str__(self):
		return(self.ufunc.template%{"arg":str(self.arg)})

#===============================================================================
class BinaryOperation(ArrayExpression):
	def __init__(self, ufunc, arg1, arg2):
		self.ufunc = ufunc
		self.arg1 = assimilate(arg1).copy()
		self.arg2 = assimilate(arg2).copy()
		if(self.arg1.dtype != self.arg2.dtype): raise(Exception("Casting is not supported, yet"))
		self.dtype = self.arg1.dtype
		self.broadcast = Broadcast(self.arg1.shape, self.arg2.shape)
		self.shape = self.broadcast.shape
		
	def build_get(self, builder, index):
		index1 = self.broadcast.index1(index)
		index2 = self.broadcast.index2(index)
		arg1_uid = self.arg1.src.build_get(builder, index1)
		arg2_uid = self.arg2.src.build_get(builder, index2)
		code = self.ufunc.template % {"arg1":arg1_uid, "arg2":arg2_uid}
		uid = builder.uid()
		builder.writeln("%s %s = %s;"%(self.dtype, uid, code))
		return(uid)
	
	def __str__(self):
		return(self.ufunc.template%{"arg1":str(self.arg1), "arg2":str(self.arg2)})
#===============================================================================
class ufunc:
	pass

class UnaryUfunc(ufunc):
	def __init__(self, template, outtype=None):
		self.template = template
		self.outtype = outtype
	
	def __call__(self, arg):
		return(UnaryOperation(self, arg))

#===============================================================================
class ReduceUfunc(ArrayExpression):
	def __init__(self, ufunc, arg, axis=0, dtype=None, out=None):
		assert(dtype==None) #not implemented, yet
		assert(out==None) #not implemented, yet
		self.ufunc = ufunc
		self.arg = assimilate(arg)
		self.axis = axis
		self.dtype = self.arg.dtype
		self.shape = self.arg.shape[:axis] + self.arg.shape[axis+1:] 
	
	def build_get(self, builder, index):
		#init
		builder.writeln("//init")
		init_uid = self.arg.src.build_get(builder, index[:self.axis]+("0",)+index[self.axis:])
		out_uid = builder.uid("out")  #loop-variable
		builder.writeln("%s %s = %s;"%(self.dtype, out_uid, init_uid))
		builder.writeln("//loop")
		n = builder.add_arg(self.arg.shape[self.axis], "N") #length of loop
		i = builder.uid("i")  #loop-variable
		builder.writeln("for (int %s=1; %s<%s; %s++) "%(i,i,n,i)) #begining at 1
		builder.writeln("{")
		a_uid = self.arg.src.build_get(builder, index[:self.axis]+(i,)+index[self.axis:])
		builder.write(out_uid+" = ")
		builder.write(self.ufunc.template%{"arg1":out_uid, "arg2":a_uid})
		builder.writeln(";")
		builder.writeln("}")
		return(out_uid)

		
		
	def __str__(self):
		return("Reduce(%s)"%str(self.arg))
	
#===============================================================================
class BinaryUfunc(ufunc):
	def __init__(self, template):
		self.template = template
	
	def __call__(self, arg1, arg2):
		return(BinaryOperation(self, arg1, arg2))
		
	def reduce(self, a, axis=0, dtype=None, out=None):
		return(ReduceUfunc(self, a, axis, dtype, out))
		
	
#===============================================================================
add = BinaryUfunc("(%(arg1)s + %(arg2)s)")
sub = BinaryUfunc("(%(arg1)s - %(arg2)s)")
div = BinaryUfunc("(%(arg1)s / %(arg2)s)")
mul = BinaryUfunc("(%(arg1)s * %(arg2)s)")
sin = UnaryUfunc("sin(%(arg)s)", numpy.dtype(numpy.float64))
square = UnaryUfunc("(%(arg)s * %(arg)s)")
sqrt = UnaryUfunc("sqrt(%(arg)s)")
def dot(a, b): return( sum(a*b, axis=0) )

#===============================================================================
class Slicing(ArrayExpression):
	def __init__(self, arg, slices):
		self.arg = assimilate(arg) #no copy - creates a view
		self.dtype = self.arg.dtype
		if(not hasattr(slices, "__iter__")):
			slices = (slices, )
			
		#replace first Ellipsis
		for (i,s) in enumerate(slices):
			if(s == Ellipsis):
				c = __builtin__.sum(1 for x in slices if x not in (None, numpy.newaxis))
				fill = (slice(None, None, None),) * (self.arg.ndim -c-i+1)
				slices = slices[:i] + fill + slices[i+1:]
				break
				
		#replace remaining Ellipsis
		for (i,s) in enumerate(slices):
			if(s == Ellipsis):
				slices = slices[:i] +(slice(None, None, None),)  + slices[i+1:]
		
		j = 0 #points to dim in self.arg.shape which we need to consume next
		out_shape = []
		for s in slices:
			if(s in (None, numpy.newaxis)):
				out_shape.append(1)  	#not incrementing j
			elif(isinstance(s, int)):
				assert(abs(s) <= self.arg.shape[j])
				j += 1
			elif(isinstance(s, slice)):
				(first, last, step) = s.indices(self.arg.shape[j])
				out_shape.append(max(0, int((last - first)/step))) #TODO: correct?
				j += 1
			else:
				raise(Exception("Strange slice: "+str(s)))
				
		out_shape += self.arg.shape[j:]
		assert(all(s>=0 for s in out_shape))
		self.shape = tuple(out_shape)
		self.slices = tuple(slices)
		
	def mk_new_index(self, builder, index):
		index = map(str, index)
		assert(len(index) == len(self.shape))
		argshape = self.arg.src.build_shape(builder)
		new_index = []
		j = 0 # were we are in index
		for s in self.slices:
			if(s in (None, numpy.newaxis)):
				assert(index[j] == "0")
				j += 1
			elif(isinstance(s, int)):
				foo = str(s)
				if(s < 0):
					foo += "+%s"%argshape[len(new_index)]
				new_index.append(foo)
			elif(isinstance(s, slice)):
				start = str(s.start)
				if(s.start==None):
					start = '0'
				elif(s.start < 0):	
					start += "+%s"%argshape[len(new_index)]
				step = s.step if(s.step!=None) else 1
				new_index.append("(%s+%d*%s)"%(start, step, index[j]))
				j += 1
			else:
				raise(Exception("Strange slice: "+str(s)))
		
		new_index += index[j:]
		return(new_index)
		
		
	def build_get(self, builder, index):
		return self.arg.src.build_get(builder, self.mk_new_index(builder, index))
	
	def build_set(self, builder, index, value):
		return self.arg.src.build_set(builder, self.mk_new_index(builder, index), value)
		
#===============================================================================
class ravel(ArrayExpression):
	""" Returns a flattened array. """
	def __init__(self, a, order='C'):
		assert(order=='C') #not implemented, yet
		self.a = assimilate(a) #no copy  - creates a view
		self.dtype = self.a.dtype
		self.shape = (self.a.size,)
		
	#def build_shape(self, builder):
	#	return( ["*".join(self.a.src.build_shape(builder))] )
	
	def build_get(self, builder, index):
		new_index = [index[0] for i in self.a.shape]
		for (i, n_uid) in enumerate( self.a.src.build_shape(builder) ):
			for j in range(i,self.a.ndim):
				new_index[j] += "%" if(i == j) else "/"
				new_index[j] += n_uid					
		return self.a.src.build_get(builder, new_index)

	def __str__(self):
		return("ravel(%s)"%str(self.a))

#===============================================================================
class zeros_numc(ArrayExpression):
	""" Returns a flattened array. """
	def __init__(self, shape, dtype=numpy.dtype(numpy.float64)):
		self.dtype = dtype
		if(isinstance(shape, int)):
			shape = (shape,)	
		self.shape = shape
		
	def build_get(self, builder, index):
		return "0"

	def __str__(self):
		return("zeros(shape=%s)"%str(self.shape))

#===============================================================================
@AssimilateDecorator
def empty(*args): 
	return(numpy.empty(*args))

@AssimilateDecorator
def zeros(*args): 
	return(numpy.zeros(*args))

#===============================================================================


#===============================================================================
def sum(a, axis=None, dtype=None, out=None):
	if(axis==None):
		a = ravel(a)
		axis = 0
	return add.reduce(a, axis, dtype, out)


#===============================================================================
def mean(a, axis=None, dtype=None, out=None):
	b = sum(a, axis, dtype, out)
	return(b / float(a.size / b.size))

#===============================================================================
def	average(a, axis=None, weights=None, returned=False):
	if(weights != None):
		a = a*weights
	b = sum(a, axis)
	sum_of_weights = sum(weights)
	average = b/float(sum_of_weights)
	if(returned):
		return((average, sum_of_weights),)
	else:
		return(sum_of_weights)
	
#===============================================================================
#TODO write more like these


#===============================================================================
#===============================================================================
# New crazy Ideas

# def propagate(propagator, start_value, times):
	# """ something like A(A(...A(A(A(start_value))))..)) """
	# proxy = Proxy()
	# proxy.set_inner = start_value
	# for i in range(times):
		# tmp = propagator(proxy)
		# proxy.set_inner = tmp
	# 



#===============================================================================
#leftovers
#class Sum(ReduceLoop):
#	operation = add


#EOF