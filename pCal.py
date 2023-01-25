from enum import Enum
import sys

class TokenType(Enum):
    KEYWORD = 0
    IDENTIFIER = 1
    INT = 2
    FLOAT = 3
    STR = 4
    ADD = 5
    SUB = 6
    MUL = 7
    DIV = 8
    MOD = 9
    EQ = 10
    LSR = 11
    GRT = 12
    LSE = 13
    GRE = 14
    NOT = 15
    NEQ = 16
    ISEQ = 17
    LPAREN = 18
    RPAREN = 19
    SEMICOL = 20
    ENDL = 21
    EOF = 22

KEYWORDS = ['cal', 'if', 'else', 'True', 'False', 'end', 'say', 'give']

class Position:
    def __init__(self, index, line, file_name, line_text):
        self.index = index
        self.line = line
        self.file_name = file_name
        self.line_text = line_text
    
    def advance(self, current_char, text):
        self.index += 1

        if current_char == '\n':
            self.line += 1
            try:
                self.line_text = text.splitlines()[self.line - 1]
            except:
                pass
    
    def duplicate(self):
        return Position(self.index, self.line, self.file_name, self.line_text)

class Token:
    def __init__(self, type : TokenType, value = None, pos : Position = None):
        self.type = type
        self.value = value
        if pos:
            self.pos = pos.duplicate()
    
    def __repr__(self):
        if self.value:
            return f'{self.type}:{self.value}'
        return f'{self.type}'
    
    def equals(self, type, value):
        return self.type == type and self.value == value

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
    
    def register(self, result):
        if result.error:
            self.error = result.error
        return result.node
    
    def success(self, node):
        self.node = node
        return self
    
    def failure(self, error):
        self.error = error
        return self

class RuntimeResult:
    def __init__(self):
        self.error = None
        self.value = None
    
    def register(self, result):
        if result.error:
            self.error = result.error
        return result.value
    
    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self

class Error:
    def __init__(self, name, info, pos : Position):
        self.name = name
        self.info = info
        self.pos = pos
    
    def __repr__(self):
        return f'File "{self.pos.file_name}" at line {self.pos.line}:\n\t"{self.pos.line_text}"\n{self.name}: {self.info}'

class IllegalCharError(Error):
    def __init__(self, info, pos):
        super().__init__('Illegal Character', info, pos)

class InvalidSyntaxError(Error):
    def __init__(self, info, pos):
        super().__init__('Invalid Syntax', info, pos)

class RuntimeError(Error):
    def __init__(self, info, pos):
        super().__init__('Runtime Error', info, pos)

class Node:
    def __init__(self, token : Token):
        self.token = token

class NumberNode(Node):
    pass

class BoolNode(Node):
    pass

class StringNode(Node):
    pass

class UnaryOperationNode(Node):
    def __init__(self, token, node):
        super().__init__(token)
        self.node = node
    
    def __repr__(self):
        return f'({self.token}, {self.node})'

class BinaryOperationNode(Node):
    def __init__(self, left, token : Token, right):
        super().__init__(token)
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f'({self.left}, {self.token}, {self.right})'

class AccessNode(Node):
    def __init__(self, name_token):
        self.name_token = name_token

class DeclarationNode(Node):
    def __init__(self, name_token, value_node = None):
        self.name_token = name_token
        self.value_node = value_node

class AssignmentNode(Node):
    def __init__(self, name_token, value_node):
        self.name_token = name_token
        self.value_node = value_node
    
    def __repr__(self):
        return f'({self.name_token} = {self.value_node})'


class InstructionsNode(Node):
    def __init__(self, instruction_nodes):
        self.instruction_nodes = instruction_nodes

class IfNode(Node):
    def __init__(self, main_case, else_case = None):
        self.main_case = main_case
        self.else_case = else_case

class SayNode(Node):
    def __init__(self, node):
        self.node = node

class InputNode(Node):
    def __init__(self):
        pass

class Context:
    def __init__(self, name):
        self.name = name
        self.symbol_table = SymbolTable()

class SymbolTable():
    def __init__(self):
        self.symbols = {}
    
    def get_var(self, name):
        value = self.symbols.get(name, None)
        return value
    
    def set_var(self, name, value):
        self.symbols[name] = value
    
    def delete_var(self, name):
        self.symbols.pop(name)

class Lexer:
    def __init__(self, expression : str, file_name : str):
        self.expression = expression
        self.pos = Position(-1, 1, file_name, "")
        self.file_name = file_name
        self.current_char = ""
        self.advance()
    
    def get_next_token(self):
        while self.pos.index < len(self.expression):
            if self.current_char == '+':
                self.advance()
                return Token(TokenType.ADD, pos=self.pos)
            elif self.current_char == '-':
                self.advance()
                return Token(TokenType.SUB, pos=self.pos)
            elif self.current_char == '*':
                self.advance()
                return Token(TokenType.MUL, pos=self.pos)
            elif self.current_char == '/':
                self.advance()
                return Token(TokenType.DIV, pos=self.pos)
            elif self.current_char == '%':
                self.advance()
                return Token(TokenType.MOD, pos=self.pos)
            elif self.current_char == '=':
                return self.get_comparison(TokenType.EQ)
            elif self.current_char == '(':
                self.advance()
                return Token(TokenType.LPAREN, pos=self.pos)
            elif self.current_char == ')':
                self.advance()
                return Token(TokenType.RPAREN, pos=self.pos)
            elif self.current_char == '"':
                return self.get_string()
            elif self.current_char == '<':
                return self.get_comparison(TokenType.LSR)
            elif self.current_char == '>':
                return self.get_comparison(TokenType.GRT)
            elif self.current_char == '!':
                return self.get_comparison(TokenType.NOT)
            elif self.current_char == ':':
                self.advance()
                return Token(TokenType.SEMICOL, pos=self.pos)
            elif self.current_char == '\n':
                self.advance()
                return Token(TokenType.ENDL, pos=self.pos)
            elif self.current_char == ' ' or self.current_char == '\t':
                pass
            else:
                if self.current_char.isnumeric():
                    return self.get_num_token()
                if self.current_char.isalpha():
                    return self.get_identifier_token()
                self.errors.append(IllegalCharError("'" + self.current_char + "'", self.pos.duplicate()))
            self.advance()
        return Token(TokenType.EOF, pos=self.pos)
    
    def lex(self):
        tokens = []
        self.errors = []
        t = self.get_next_token()
        while t.type !=  TokenType.EOF:
            tokens.append(t)
            t = self.get_next_token()
        tokens.append(t)
        return tokens

    def get_num_token(self):
        value = ""
        dot_num = 0
        start_pos = self.pos.duplicate()
        while self.pos.index < len(self.expression) and (self.current_char.isnumeric() or self.current_char == '.'):
            if self.current_char == '.':
                if dot_num == 1: break
                dot_num += 1
                value += '.'
            else:
                value += self.current_char
            
            self.advance()
        
        if dot_num == 0:
            return Token(TokenType.INT, value, pos=start_pos)
        return Token(TokenType.FLOAT, value, pos=start_pos)

    def get_identifier_token(self):
        value = ""
        start_pos = self.pos.duplicate()
        while self.pos.index < len(self.expression) and (self.current_char.isalpha() or self.current_char.isnumeric() or self.current_char == '_'):
            value += self.current_char
            self.advance()
        
        token_type = TokenType.KEYWORD if value in KEYWORDS else TokenType.IDENTIFIER
        return Token(token_type, value, pos=start_pos)
    
    def get_string(self):
        value = ""
        start_pos = self.pos.duplicate()
        escape_characters = {'n': '\n',
                             't': '\t'}
        self.advance()
        escape_character = False

        while self.current_char != '"':
            if self.current_char == None:
                self.errors.append(InvalidSyntaxError("Expected '\"'", start_pos))
                return Token(TokenType.STR, pos=start_pos)
            if escape_character:
                value += escape_characters.get(self.current_char, self.current_char)
                escape_character = False
            else:
                if self.current_char == "\\":
                    escape_character = True
                else:
                    value += self.current_char
            self.advance()
        
        self.advance()
        return Token(TokenType.STR, value, pos=start_pos)
    
    def get_comparison(self, token_type):
        start_pos = self.pos.duplicate()
        self.advance()
        if self.current_char == "=":
            self.advance()
            if (token_type == TokenType.LSR):
                return Token(TokenType.LSE, pos=start_pos)
            elif (token_type == TokenType.GRT):
                return Token(TokenType.GRE, pos=start_pos)
            elif (token_type == TokenType.NOT):
                return Token(TokenType.NEQ, pos=start_pos)
            elif (token_type == TokenType.EQ):
                return Token(TokenType.ISEQ, pos=start_pos)

        return Token(token_type, pos=start_pos)
    
    def advance(self):
        self.pos.advance(self.current_char, self.expression)
        self.current_char = self.expression[self.pos.index] if self.pos.index < len(self.expression) else None

class Parser:
    # instructions : "\n"* instr ("\n"+ instr)* "\n"*
    # instr : declaration | assignment | comp-expr | say-instr
    # declaration : "cal" IDENTIFIER ("=" comp-expr)?
    # assignment : IDENTIFIER "=" comp-expr
    # say-instr : "say" "(" comp-expr ")"
    # comp-expr : math-expr ((LSR|LSE|GRT|GRE|NEQ) math-expr)* | NOT comp-expr
    # math-expr   : term ((ADD|SUB) term)*
    # term   : factor ((MUL|DIV|MOD) factor)*
    # factor : NUM | BOOL | STR | IDENTIFIER | input | (ADD|SUB) factor | "(" comp-expr ")" | if-stat
    # input : "<" give ">"
    # if-stat : "if" comp-expr ":" ("\n" instructions) "end"

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
    
    def parse(self):
        result = ParseResult()
        instructions = result.register(self.instructions())
        if result.error:
            return result
        if (self.tokens[self.pos].type != TokenType.EOF):
            return result.failure(InvalidSyntaxError("Unrecognized grammar", self.tokens[self.pos].pos))
        return result.success(instructions)
    
    def factor(self):
        result = ParseResult()
        t = self.tokens[self.pos]
        if t.type in (TokenType.ADD, TokenType.SUB):
            self.advance()
            node = result.register(self.factor())
            if result.error:
                return result
            return result.success(UnaryOperationNode(t, node))
        
        elif t.type == TokenType.INT or t.type == TokenType.FLOAT:
            self.advance()
            return result.success(NumberNode(t))
        
        elif t.equals(TokenType.KEYWORD, "True") or t.equals(TokenType.KEYWORD, "False"):
            self.advance()
            return result.success(BoolNode(t))
        
        elif t.type == TokenType.STR:
            self.advance()
            return result.success(StringNode(t))
        
        elif t.type == TokenType.LSR:
            input_expr = result.register(self.input_expr())
            if result.error:
                return result
            return result.success(input_expr)

        elif t.type == TokenType.LPAREN:
            self.advance()
            comp_expr = result.register(self.comp_expr())
            if result.error:
                return result
            if self.tokens[self.pos].type == TokenType.RPAREN:
                self.advance()
                return result.success(comp_expr)
            return result.failure(InvalidSyntaxError("Expected ')'", t.pos))
        
        elif t.type == TokenType.IDENTIFIER:
            self.advance()
            return result.success(AccessNode(t))
        
        elif t.equals(TokenType.KEYWORD, "if"):
            self.advance()
            if_statement = result.register(self.if_statement())
            if result.error:
                return result
            return result.success(if_statement)
        
        elif t.equals(TokenType.KEYWORD, "end"):
            return result.success(Node(Token(TokenType.KEYWORD, pos=t.pos)))

        return result.failure(InvalidSyntaxError("Expected a number or identifier", t.pos))
    
    def term(self):
        return self.do_operation(self.factor, (TokenType.MUL, TokenType.DIV, TokenType.MOD))
    
    def math_expr(self):
        return self.do_operation(self.term, (TokenType.ADD, TokenType.SUB))
    
    def comp_expr(self):
        result = ParseResult()
        if self.tokens[self.pos].type == TokenType.NOT:
            op_token = self.tokens[self.pos]
            self.advance()
            node = result.register(self.comp_expr())
            if result.error:
                return result
            return result.success(UnaryOperationNode(op_token, node))
        
        node = result.register(self.do_operation(self.math_expr, (TokenType.LSR, TokenType.LSE, TokenType.GRT, TokenType.GRE, TokenType.NEQ, TokenType.ISEQ)))
        if result.error:
            return result
        
        return result.success(node)
    
    def say_instr(self):
        result = ParseResult()
        self.advance()

        if (self.tokens[self.pos].type == TokenType.LPAREN):
            self.advance()
            comp_expr = result.register(self.comp_expr())
            if result.error:
                return result
            if (self.tokens[self.pos].type == TokenType.RPAREN):
                self.advance()
                return result.success(SayNode(comp_expr))
                
            return result.failure(InvalidSyntaxError('Expected ")"', self.tokens[self.pos].pos))

        return result.failure(InvalidSyntaxError('Expected "("', self.tokens[self.pos].pos))

    def assignment(self):
        pass

    def declaration(self):
        result = ParseResult()
        self.advance()

        if self.tokens[self.pos].type == TokenType.IDENTIFIER:
            var_name_token = self.tokens[self.pos]
            self.advance()
            if self.tokens[self.pos].type == TokenType.EQ:
                self.advance()
                comp_expr = result.register(self.comp_expr())
                if result.error:
                    return result
                return result.success(DeclarationNode(var_name_token, comp_expr))
            else:
                return result.success(DeclarationNode(var_name_token))
        return result.failure(InvalidSyntaxError("Expected identifier", self.tokens[self.pos].pos))
    
    def instr(self):
        if self.tokens[self.pos].equals(TokenType.KEYWORD, "cal"):
            return self.assignment()
        elif self.tokens[self.pos].equals(TokenType.KEYWORD, "say"):
            return self.say_instr()
        return self.comp_expr()
    
    def instructions(self):
        result = ParseResult()
        instructions = []
        while self.tokens[self.pos].type == TokenType.ENDL:
            self.advance()
        if self.tokens[self.pos].type != TokenType.EOF:
            instr = result.register(self.instr())
            if result.error:
                return result
            instructions.append(instr)
        while self.tokens[self.pos].type == TokenType.ENDL:
            self.advance()
            if self.tokens[self.pos].type != TokenType.EOF and self.tokens[self.pos].type != TokenType.ENDL:
                instr = result.register(self.instr())
                if result.error:
                    return result
                instructions.append(instr)
        return result.success(InstructionsNode(instructions))
    
    def if_statement(self):
        result = ParseResult()

        comp_expr = result.register(self.comp_expr())
        if result.error:
            return result
        
        if self.tokens[self.pos].type != TokenType.SEMICOL:
            return result.failure(InvalidSyntaxError("Expected ':'", self.tokens[self.pos].pos))
        
        self.advance()
        if self.tokens[self.pos].type == TokenType.ENDL:
            self.advance()
            instructions = result.register(self.instructions())
            if result.error:
                return result
            
            if self.tokens[self.pos].equals(TokenType.KEYWORD, "end"):
                self.advance()
                return result.success(IfNode((comp_expr, instructions)))
            
            return result.failure(InvalidSyntaxError("Expected \"end\"", self.tokens[self.pos].pos))
        
        return result.failure(InvalidSyntaxError("Expected new line", self.tokens[self.pos].pos))
    
    def input_expr(self):
        result = ParseResult()
        self.advance()
        if self.tokens[self.pos].equals(TokenType.KEYWORD, "give"):
            self.advance()
            if self.tokens[self.pos].type == TokenType.GRT:
                self.advance()
                return result.success(InputNode())
            return result.failure(InvalidSyntaxError('Expected "<"', self.tokens[self.pos].pos))
        return result.failure(InvalidSyntaxError('Expected "give"', self.tokens[self.pos].pos))
    
    def do_operation(self, func, operations):
        result = ParseResult()
        left = result.register(func())
        if result.error:
            return result

        while self.tokens[self.pos].type in operations:
            op_token = self.tokens[self.pos]
            self.advance()
            right = result.register(func())
            if result.error:
                return result
            left = BinaryOperationNode(left, op_token, right)
        
        return result.success(left)

    def advance(self):
        self.pos += 1
        if self.pos >= len(self.tokens):
            self.pos -= 1

class Interpreter: 
    def evaluate(self, node, context):
        result = RuntimeResult()
        if isinstance(node, InstructionsNode):
            instructions = []
            for instruction_node in node.instruction_nodes:
                value = result.register(self.evaluate(instruction_node, context))
                if value != None:
                    instructions.append(value)
                if result.error:
                    return result
            return result.success(instructions)
        elif isinstance(node, NumberNode):
            return result.success(self.int_or_float(node))
        elif isinstance(node, BoolNode):
            return result.success(True if node.token.value == "True" else False)
        elif isinstance(node, StringNode):
            return result.success(node.token.value)
        elif isinstance(node, BinaryOperationNode):
            left = result.register(self.evaluate(node.left, context))
            if result.error:
                return result
            right = result.register(self.evaluate(node.right, context))
            if result.error:
                return result
            try:
                left + right
            except:
                return result.failure(RuntimeError(f"Can not do operation on  and ", node.token.pos))
            if node.token.type == TokenType.ADD:
                return result.success(left + right)
            elif node.token.type == TokenType.SUB:
                return result.success(left - right)
            elif node.token.type == TokenType.MUL:
                return result.success(left * right)
            elif node.token.type == TokenType.DIV:
                if right == 0:
                    return result.failure(RuntimeError("Division by 0", node.token.pos))
                return result.success(left / right)
            elif node.token.type == TokenType.MOD:
                if right == 0:
                    return result.failure(RuntimeError("Division by 0", node.token.pos))
                return result.success(left % right)
            elif node.token.type == TokenType.LSR:
                return result.success(left < right)
            elif node.token.type == TokenType.GRT:
                return result.success(left > right)
            elif node.token.type == TokenType.GRE:
                return result.success(left >= right)
            elif node.token.type == TokenType.LSE:
                return result.success(left <= right)
            elif node.token.type == TokenType.NEQ:
                return result.success(left != right)
            elif node.token.type == TokenType.ISEQ:
                return result.success(left == right)
        elif isinstance(node, UnaryOperationNode):
            value = result.register(self.evaluate(node.node, context))
            if result.error:
                return result
            if node.token.type == TokenType.SUB:
                value *= (-1)
            elif node.token.type == TokenType.NOT:
                value = not value
            return result.success(value)
        elif isinstance(node, DeclarationNode):
            name = node.name_token.value
            if context.symbol_table.get_var(name):
                result.failure(RuntimeError(f'Variable "{name}" is already defined', node.name_token.pos))
            value = result.register(self.evaluate(node.value_node, context))
            if result.error:
                return result
            context.symbol_table.set_var(name, value)
            return result.success(value)
        elif isinstance(node, AssignmentNode):
            name = node.name_token.value
            if not context.symbol_table.get_var(name):
                result.failure(RuntimeError(f'Variable "{name}" is not defined', node.name_token.pos))
            value = result.register(self.evaluate(node.value_node, context))
            if result.error:
                return result
            context.symbol_table.set_var(name, value)
            return result.success(value)
        elif isinstance(node, AccessNode):
            name = node.name_token.value
            value = context.symbol_table.get_var(name)
            if value != None:
                return result.success(value)
            return result.failure(RuntimeError(f'Variable "{name}" is not defined', node.name_token.pos))
        elif isinstance(node, IfNode):
            condition, instructions = node.main_case
            condition = result.register(self.evaluate(condition, context))
            if result.error:
                return result
            if condition:
                instructions = result.register(self.evaluate(instructions, context))
                if result.error:
                    return result
                return result.success(instructions)
            return result.success(None)
        elif isinstance(node, SayNode):
            value = result.register(self.evaluate(node.node, context))
            if result.error:
                return result
            print(value, end='')
            return result.success(value)
        elif isinstance(node, InputNode):
            value = input()
            value = self.validate_input(value)
            return result.success(value)
        elif isinstance(node, Node):
            return result.success(None)
    
    def int_or_float(self, node):
        if node.token.type == TokenType.INT:
            return int(node.token.value)
        elif node.token.type == TokenType.FLOAT:
            return float(node.token.value)
        return node.token.value
    
    def validate_input(self, value):
        try:
            return int(value)
        except:
            pass
        try:
            return float(value)
        except:
            pass
        try:
            return bool(value)
        except:
            pass
        return value

def execute(file_name):
    context = Context("<program>")
    try:
        file = open(file_name, "r")
    except IOError:
        print(f"Error: File \"{file_name}\" does not exist.")
        return
    lexer = Lexer(file.read(), file_name)
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
                pass
    print()

if len(sys.argv) > 1:
    execute(sys.argv[1])