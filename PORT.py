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
events=FileInputStream(test_args[1])
cep.run(events,FileOutputStream("port/outputFiles",test_args[2]),SysCallDataFormatter())

#cep=CEP([port.patterns.kleene_closure1])


#events=FileInputStream("port/eventFiles/kleene_closure1.txt")

#cep.run(events,FileOutputStream("port/outputFiles","output.txt"),SysCallDataFormatter())
