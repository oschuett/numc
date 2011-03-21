from scipy import weave
import sys
import itertools
import numpy
import traceback
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
			result = ndarray(NumpyArray(result))
		return result



sys.modules[__name__] = ModuleWrapper(sys.modules[__name__])

			
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
	return( ndarray(NumpyArray(something)) )

#===============================================================================
class ArraySource(object):
	""" The Base-Class """
	pass


#===============================================================================
class ArrayExpression(ArraySource):
	def __new__(cls, *args, **kwargs):
		#print("__new__ called: "+cls.__name__)
		new_obj = object.__new__(cls)
		cls.__init__(new_obj, *args, **kwargs)
		return ndarray(new_obj)
	
		
	def evaluate(self):
		""" Evaluate itself for all indices. Results are accessable via __array_interface__ """
		print("evaluating: %s"%self)
		B = CodeBuilder()
		index = B.loop(self)
		B.writeln("{")
		a_uid = self.build(B, index)
		out = empty(self.shape, self.dtype)
		out_uid = out.src.build(B, index)
		B.writeln("%s = %s;"%(out_uid, a_uid))
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
		#reduce(operator.mul, self.shape)
		return( numpy.prod(self.shape) )
	
	
	def __add__(self, other): return( add(self, other) )
	def __sub__(self, other): return( sub(self, other) )
	def __div__(self, other): return( div(self, other) )
	
	
	def __getitem__(self, slices):
		print "__getitem__(%s) called"%str(slices)
		return( Slice(self, slices) )
		
		
	def __setitem__(self, key, value):
		print "__setitem__(%s, %s) called"%(key, value)
		if(not isinstance(self.src, NumpyArray)):
			self.src = self.src.evaluate().src
		#print("Refcount: %d"%sys.getrefcount(self.src))
		if(sys.getrefcount(self.src) > 2): #one for self.src and one for getrefcount()
			print "Makeing a copy!!!!!!!!!"
			self.src = NumpyArray(self.src.array.copy())
		self.src.array.__setitem__(key, value)
	
	
		
	@property
	def __array_interface__(self):
		try:
			# also called when ndarray gets passed to numpy-functions 
			#traceback.print_stack()
			#TODO is this a possible write-access?
			#print("Array interface called: %s"%self)
			#print("Array interface called")
			if(not isinstance(self.src, NumpyArray)):
				self.src = self.src.evaluate().src
				#print("Refcount: %d"%sys.getrefcount(self.src))
			return(self.src.array.__array_interface__)
		except:
			print("!!!!!!!!!! An exception in __array_interface__ occured !!!!!!!!!!!!!")
			traceback.print_exc()
	
	def __str__(self):
		return(str(self.src))


#===============================================================================
class NumpyArray(ArraySource):
	""" Wrapper to handle e.g. numpy.ndarray objects transparently. """
	#TODO: support any thing that porvides __array_interface__
	def __init__(self, array):
		assert(isinstance(array, numpy.ndarray))
		self.array = array
		self.shape = array.shape
		self.dtype = array.dtype 
		
	def build(self, builder, index):
		arg_uid = self._add2builder(builder)
		if(len(index) == 0):
			return("*"+arg_uid)
		index_code = index[0]
		for (n,i) in enumerate(index[1:]):
			index_code = "( %s ) * %s_shape_%d + %s"%(index_code, arg_uid, n+1, i)
		return("%s[%s]"%(arg_uid, index_code))

	def build_shape(self, builder): #TODO: improve
		self_uid = self._add2builder(builder)
		return [self_uid+"_shape_"+str(i) for i in range(len(self.shape))]
	
	def _add2builder(self, builder):
		if(id(self) not in builder.cache.keys()):
			self_uid = builder.add_arg(self.array)
			for (i,n) in enumerate(self.array.shape):
				builder.add_arg(n, self_uid+"_shape_"+str(i))
			builder.cache[id(self)] = self_uid
		return( builder.cache[id(self)] )
	
	def __str__(self):
		return(str(self.array))
		#return("NumpyArray%s"%str(self.array.shape))

#===============================================================================
class CodeBuilder():
	""" Centerpiece during generation of C-Code """
	def __init__(self):
		self.code = ""
		self.args = {}
		self.uids = set()
		self.cache = {}
		
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
					return(v) 
		uid = self.uid(name)
		self.args[uid] = arg
		return(uid)
			
	def write(self, code):
		self.code += code

	def writeln(self, code):
		self.write(code+"\n")

	def loop(self, arg):
		index = []
		for n in arg.build_shape(self):
			#n = self.add_arg(N, "N") #length of loop
			i = self.uid("i")  #loop-variable
			index.append(i)
			self.writeln("for (int %s=0; %s<%s; %s++) "%(i,i,n,i))
		return(index)
	
	def run(self):
		self.code = self.code.replace("float", "float64") #TODO: solve genericly
		self.code = self.code.replace("float64", "double") #TODO: solve genericly
		self.code = self.code.replace("double64", "double") #TODO: solve genericly
		self.code = self.code.replace("int32", "int") #TODO: solve genericly
		verbose = 2
		#verbose = 0
		print("Running C-Code...")
		if(verbose > 0):
			print "Running:\n"+ self.code
			print self.args.keys()
		weave.inline(self.code, self.args.keys(), self.args,
				force=False, verbose=verbose)
					#type_converters=weave.converters.blitz, compiler = 'gcc', verbose=2)


	
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
#TODO: evtl via Metaclass
def HandlefyDecorator(inner_func):
	def deco_func(*args, **kwargs):
		result = inner_func(*args, **kwargs)
		#TODO: result could be iterable
		if(isinstance(result, numpy.ndarray)):
			result = NumpyArray(result)
		if(isinstance(result, ArraySource)):
			result = ndarray(result)
		return(result)
	
	return(deco_func)

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
		
	def build(self, builder, index):
		arg_uid = self.arg.src.build(builder, index)
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
		

	def build(self, builder, index):
		index1 = self.broadcast.index1(index)
		index2 = self.broadcast.index2(index)
		print type(self)
		print type(self.arg1)
		print type(self.arg1.src)
		arg1_uid = self.arg1.src.build(builder, index1)
		arg2_uid = self.arg2.src.build(builder, index2)
		code = self.ufunc.template % {"arg1":arg1_uid, "arg2":arg2_uid}
		uid = builder.uid()
		builder.writeln("%s %s = %s;"%(self.dtype, uid, code))
		return(uid)
	
	def build_shape(self, builder): #TODO: improve
		return [builder.add_arg(n) for n in self.shape]
	
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
class BinaryUfunc(ufunc):
	def __init__(self, template):
		self.template = template
	
	def __call__(self, arg1, arg2):
		return(BinaryOperation(self, arg1, arg2))
		
	
	def reduce(self, a, axis=0, dtype=None, out=None):
		if(not isinstance(a, ndarray)):
			raise(Exception("not an ndarray - should forward to NumPy - not yet Implemented."))
			#print("reduce: Not an ArrayExpression - forwarding to NumPy")
			#return(numpy.sum(a, axis, dtype, out))
		if(not dtype): dtype = a.dtype
		
		out_shape = a.shape[:axis] + a.shape[axis+1:] 
			
		if(out):
			assert(out.shape == out_shape)
			assert(out.dtype == dtype) #TODO:sure?
			out[:] = 0.0 
		else:
			out = zeros(out_shape, dtype)
		#print "out: %s"%type(out)
		#print "out: %s"%repr(out)
		B = CodeBuilder()
		index = B.loop(a.src)
		#print index
		B.writeln("{")
		
		out_uid = out.src.build(B, index[:axis]+index[axis+1:])
		tmp_uid = B.uid("tmp")
		a_uid = a.src.build(B, index)
		B.writeln("%s %s = %s;"%(a.dtype, tmp_uid,a_uid))
		B.write(out_uid+" = ")
		B.write(self.template%{"arg1":out_uid, "arg2":tmp_uid})
		B.writeln(";")
		
		B.writeln("}")
		B.run()  #compile and run code
		return(out)

add = BinaryUfunc("( %(arg1)s + %(arg2)s )")
sub = BinaryUfunc("( %(arg1)s - %(arg2)s )")
div = BinaryUfunc("( %(arg1)s / %(arg2)s )")
sin = UnaryUfunc("sin(%(arg)s)", numpy.dtype(numpy.float64))
square = UnaryUfunc("( %(arg)s * %(arg)s)")


#===============================================================================
class Slice(ArrayExpression):
	def __init__(self, arg, slices):
		self.arg = assimilate(arg) #no copy - creates a view
		print slices
		assert(False)
		# self.arg = arg # not incrementing refcount - it's only a view
		# self.key = key
		# self.shape = []
		# for part in key:
			# print dir(part)
			# if(isinstance(part, slice)):
				# if(part.start==None and part.start==None and part.start==None):
					# self.shape.append(						
				# 
				# print part.indices()
		# print key
		# 

#===============================================================================
class ravel(ArrayExpression):
	""" Returns a flattened array. """
	def __init__(self, a, order='C'):
		assert(order=='C') #not implemented, yet
		self.a = assimilate(a) #no copy  - creates a view
		self.dtype = self.a.dtype
		self.shape = (self.a.size,)
		
	def build_shape(self, builder):
		return( ["*".join(self.a.src.build_shape(builder))] )
	
	def build(self, builder, index):
		new_index = [index[0] for i in self.a.shape]
		for (i, n_uid) in enumerate( self.a.src.build_shape(builder) ):
			for j in range(i,self.a.ndim):
				new_index[j] += "%" if(i == j) else "/"
				new_index[j] += n_uid					
		return self.a.src.build(builder, new_index)



#===============================================================================
#TODO: implement as ArrayExpression , see below
@HandlefyDecorator
def arange(*args): return(numpy.arange(*args))

@HandlefyDecorator
def zeros(*args): return(numpy.zeros(*args))

@HandlefyDecorator
def empty(*args): return(numpy.empty(*args))

#===============================================================================
# Produces: 
# 0.0 = ( 0.0 + tmp );
#TODO: need to distinguis betwen build_get and build_set 
# class zeros(ArrayExpression):
	# def __init__(self, shape, dtype=numpy.dtype(numpy.float64)): #other args not supported, yet
		# self.shape = shape
		# self.dtype = dtype
		# assert(str(dtype) == "float64")
	# 
	# def build_shape(self, builder):
		# return(builder.add_arg(s) for s in self.shape)
		# 
	# def build(self, builder, index):
		# return("0.0")

#===============================================================================
@HandlefyDecorator
def sum(a, axis=None, dtype=None, out=None):
	if(axis==None):
		a = ravel(a) #let ravel to the assimilate
		axis = 0
	return add.reduce(a, axis, dtype, out)

#===============================================================================
# class sum(ndarray):
	# def __init__(self, a, axis=None, dtype=None, out=None):
		# if(axis==None):
			# a = ravel(a) #let ravel to the assimilate
			# axis = 0
		# result = add.reduce(a, axis, dtype, out)
		# ndarray.__init__(self, result.src) 
	# 
#===============================================================================
@HandlefyDecorator
def mean(a, axis=None, dtype=None, out=None):
	b = sum(a, axis, dtype, out)
	return(b / float(a.size / b.size))

#===============================================================================
@HandlefyDecorator
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