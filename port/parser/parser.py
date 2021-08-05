import sys
from antlr4 import *
from PortLexer import PortLexer
from PortParser import PortParser
from encode_events import RewriteListener

def main(argv):
    input = FileStream(argv[1])
    lexer = PortLexer(input)
    stream = CommonTokenStream(lexer)
    parser = PortParser(stream)
    tree = parser.program()
    print(tree.toStringTree(recog=parser))

    walker = ParseTreeWalker()
    walker.walk(RewriteListener(), tree)

if __name__ == '__main__':
    main(sys.argv)