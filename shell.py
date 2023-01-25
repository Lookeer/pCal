from pCal import Lexer, Parser, Interpreter, Context

context = Context("<program>")
while True:
    user_input = input("pCal>")
    lexer = Lexer(user_input, "shell")
    tokens = lexer.lex()
    if len(lexer.errors) > 0:
        for e in lexer.errors:
            print(e)
    else:
        parser = Parser(tokens)
        ast = parser.parse()
        
        if ast.error:
            print(ast.error)
        else:
            interpreter = Interpreter()
            result = interpreter.evaluate(ast.node, context)
            if (result.error):
                print(result.error)
            else:
                print()