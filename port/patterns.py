# from test.testUtils import *
from datetime import timedelta
from condition.Condition import Variable, TrueCondition, BinaryCondition, SimpleCondition
from condition.CompositeCondition import AndCondition
from condition.BaseRelationCondition import NotEqCondition, EqCondition, GreaterThanCondition, GreaterThanEqCondition, \
    SmallerThanEqCondition
from base.PatternStructure import NegationOperator, AndOperator, SeqOperator, PrimitiveEventStructure, KleeneClosureOperator
from base.Pattern import Pattern
# from port.dataFormats.syscall import SysCallDataFormatter

kleene_closure1 = Pattern(
    SeqOperator(
        PrimitiveEventStructure("blue", "a"),
        KleeneClosureOperator(PrimitiveEventStructure("red", "b"))
    ),
    TrueCondition(),
    timedelta(minutes=5)
)

# openat and close on same file descriptor with no intervening read
negation1 = Pattern(
	SeqOperator(PrimitiveEventStructure("openat", "a"),
		        NegationOperator(PrimitiveEventStructure("read","b")),
		        PrimitiveEventStructure("close","c")),
	AndCondition(EqCondition(Variable("a", lambda x: x["File Handle"]),
                            Variable("c", lambda x: x["File Handle"])),
		        EqCondition(Variable("c", lambda x: x["File Handle"]),
                            Variable("b", lambda x: x["File Handle"]))),
    timedelta(minutes=10))