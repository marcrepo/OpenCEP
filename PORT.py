from test.testUtils import *
from datetime import timedelta
from condition.Condition import Variable, TrueCondition, BinaryCondition, SimpleCondition
from condition.CompositeCondition import AndCondition
from condition.BaseRelationCondition import NotEqCondition, EqCondition, GreaterThanCondition, GreaterThanEqCondition, \
    SmallerThanEqCondition
from base.PatternStructure import NegationOperator, AndOperator, SeqOperator, PrimitiveEventStructure, KleeneClosureOperator
from base.Pattern import Pattern
from port.dataFormats.syscall import SysCallDataFormatter
from CEP import CEP
import port.patterns

# test names 
# kleene_closure1
# negation1

test_name="kleene_closure1"

def form_test_args(test_name):
	pattern=eval("port.patterns."+test_name)
	input_stream="port/eventFiles/"+test_name+".txt"
	output_stream=test_name+".txt"
	return (pattern,input_stream,output_stream)

test_args=form_test_args(test_name)
cep=CEP([test_args[0]])
input_stream=FileInputStream(test_args[1])

# to dump matches to a file
# cep.run(input_stream,FileOutputStream("port/outputFiles",test_args[2]),SysCallDataFormatter())

# dump matches to an output stream that we can read
output_stream=OutputStream()
cep.run(input_stream,output_stream,SysCallDataFormatter())

for x in output_stream:
	print("---Start Match---")
	for y in x.events:
		print(y)
	print("---End Match---\n")