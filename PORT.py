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


# Find an openat followed by a close with no intervening read all on same file handle


pattern = Pattern(
	SeqOperator(PrimitiveEventStructure("openat", "a"),
		    NegationOperator(PrimitiveEventStructure("read","b")),
		    PrimitiveEventStructure("close","c")),
	AndCondition(EqCondition(Variable("a", lambda x: x["File Handle"]),
Variable("c", lambda x: x["File Handle"])),
		     EqCondition(Variable("c", lambda x: x["File Handle"]),
Variable("b", lambda x: x["File Handle"]))),
        timedelta(minutes=10))


cep=CEP([pattern])

events=FileInputStream("port/eventFiles/sea_test2.txt")

cep.run(events,FileOutputStream("port/outputFiles","output.txt"),SysCallDataFormatter())
