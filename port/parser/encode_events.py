from PortListener import PortListener

import sys

sys.path.append('../..')   # so we can import condition and base modules

# from patterns.py (start)
from stream.Stream import OutputStream
from stream.FileStream import FileInputStream
from datetime import timedelta
from condition.Condition import Variable, TrueCondition, BinaryCondition, SimpleCondition
from condition.CompositeCondition import AndCondition
from condition.BaseRelationCondition import NotEqCondition, EqCondition, GreaterThanCondition, GreaterThanEqCondition, \
    SmallerThanEqCondition
from base.PatternStructure import NegationOperator, AndOperator, SeqOperator, PrimitiveEventStructure, KleeneClosureOperator
from base.Pattern import Pattern
from port.dataFormats.syscall import SysCallDataFormatter
from CEP import CEP
# from patterns.py (end)

sys.path.remove("../..")  # no longer need the addition to the path for base and condition

class RewriteListener(PortListener):
    event_list=[]
    dataNamePrefix="dn"

    def _nextDataName_(self):
        return RewriteListener.dataNamePrefix+str(len(RewriteListener.event_list))

    # Enter a parse tree produced by ArrayInitParser#init.
    def enterProgram(self, ctx):
        print("Start Parsing")

    # Exit a parse tree produced by ArrayInitParser#init.
    def exitProgram(self, ctx):
        print("End Parsing")
        print(RewriteListener.event_list)
        pattern=Pattern(
            SeqOperator(*RewriteListener.event_list),
            TrueCondition(),
            timedelta(minutes=5)
        )
        print("pattern="+str(pattern))
        cep=CEP([pattern])
        input_stream=FileInputStream("event_list.txt")
        output_stream=OutputStream()
        cep.run(input_stream,output_stream,SysCallDataFormatter())

        for x in output_stream:
	        print("---Start Match---")
	        for y in x.events:
		        print(y)
	        print("---End Match---\n")

    # Enter a parse tree produced by ArrayInitParser#value.
    def enterEvent(self, ctx):
        pass

    # Exit a parse tree produced by ArrayInitParser#value.
    def exitEvent(self, ctx):
        data = ctx.ID().getText()
        dataName=self._nextDataName_()
        RewriteListener.event_list.append(PrimitiveEventStructure(data, dataName))
        print("event ID="+data+" "+dataName)
