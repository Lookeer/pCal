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
    LPAREN = 17
    RPAREN = 18
    ENDL = 19
    END = 20

KEYWORDS = ['cal', 'if']

class Position:
    def __init__(self, index, line, file_name):
        self.index = index
        self.line = line
        self.file_name = file_name
    
    def advance(self, current_char):
        self.index += 1

        if current_char == '\n':
            self.line += 1
    
    def duplicate(self):
        return Position(self.index, self.line, self.file_name)

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
        return f'File "{self.pos.file_name}" at line {self.pos.line}:\n  {self.name}: {self.info}'

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
    
    def __repr__(self):
        return f'{self.token}'

class NumberNode(Node):
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
    def __init__(self, condition_node, else_node = None):
        self.condition_node = condition_node
        self.else_node = else_node

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
        self.pos = Position(0, 1, file_name)
        self.file_name = file_name
        self.current_char = expression[0]
    
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
                self.advance()
                return Token(TokenType.EQ, pos=self.pos)
            elif self.current_char == '(':
                self.advance()
                return Token(TokenType.LPAREN, pos=self.pos)
            elif self.current_char == ')':
                self.advance()
                return Token(TokenType.RPAREN, pos=self.pos)
            elif self.current_char == '<':
                return self.get_comparison(TokenType.LSR)
            elif self.current_char == '>':
                return self.get_comparison(TokenType.GRT)
            elif self.current_char == '!':
                return self.get_comparison(TokenType.NOT)
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
        return Token(TokenType.END)
    
    def lex(self):
        tokens = []
        self.errors = []
        t = self.get_next_token()
        while t.type !=  TokenType.END:
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
        return Token(token_type, pos=start_pos)
    
    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.expression[self.pos.index] if self.pos.index < len(self.expression) else None

class Parser:
    # instructions : "\n"* instr ("\n"+ instr)* "\n"*
    # instr : assignment | comp-expr
    # assignment : "cal" IDENTIFIER "=" math-expr | IDENTIFIER "=" math-expr
    # if-expr : "if" comp-expr "{"
    # comp-expr : math-expr ((LSR|LSE|GRT|GRE|NEQ) math-expr)* | NOT comp-expr
    # math-expr   : term ((ADD|SUB) term)*
    # term   : factor ((MUL|DIV|MOD) factor)*
    # factor : NUM | IDENTIFIER | (ADD|SUB) factor | "(" comp-expr ")"

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
    
    def parse(self):
        result = ParseResult()
        instructions = self.instructions()
        if (self.tokens[self.pos].type != TokenType.END):
            return result.failure(InvalidSyntaxError("Not good", self.tokens[self.pos].pos))
        return instructions
    
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
        
        node = result.register(self.do_operation(self.math_expr, (TokenType.LSR, TokenType.LSE, TokenType.GRT, TokenType.GRE, TokenType.NEQ)))
        if result.error:
            return result
        
        return result.success(node)

    def assignment(self):
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
                return result.success(AssignmentNode(var_name_token, comp_expr))
        return result.failure(InvalidSyntaxError("Expected identifier", self.tokens[self.pos].pos))
    
    def instr(self):
        if self.tokens[self.pos].equals(TokenType.KEYWORD, "cal"):
            return self.assignment()
        return self.comp_expr()
    
    def instructions(self):
        result = ParseResult()
        instructions = []
        while self.tokens[self.pos].type == TokenType.ENDL:
            self.advance()
        if self.tokens[self.pos].type != TokenType.END:
            instr = result.register(self.instr())
            if result.error:
                return result
            instructions.append(instr)
        while self.tokens[self.pos].type == TokenType.ENDL:
            self.advance()
            if self.tokens[self.pos].type != TokenType.END:
                instr = result.register(self.instr())
                if result.error:
                    return result
                instructions.append(instr)
        return result.success(InstructionsNode(instructions))
    
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
                instructions.append(result.register(self.evaluate(instruction_node, context)))
                if result.error:
                    return result
            return result.success(instructions)
        elif isinstance(node, NumberNode):
            return result.success(self.int_or_float(node))
        elif isinstance(node, BinaryOperationNode):
            left = result.register(self.evaluate(node.left, context))
            if result.error:
                return result
            right = result.register(self.evaluate(node.right, context))
            if result.error:
                return result
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
        elif isinstance(node, UnaryOperationNode):
            value = result.register(self.evaluate(node.node, context))
            if result.error:
                return result
            if node.token.type == TokenType.SUB:
                value *= (-1)
            elif node.token.type == TokenType.NOT:
                value = not value
            return result.success(value)
        elif isinstance(node, AssignmentNode):
            name = node.name_token.value
            if context.symbol_table.get_var(name):
                pass
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
            condition = result.register(self.evaluate(node.condition_node, context))
            if result.error:
                return result
            if condition:
                pass
    
    def int_or_float(self, node):
        if node.token.type == TokenType.INT:
            return int(node.token.value)
        elif node.token.type == TokenType.FLOAT:
            return float(node.token.value)

def execute(file_name):
    context = Context("<program>")
    try:
        file = open(file_name, "r")
    except IOError:
        print(f"Error: File \"{file_name}\" does not exist.")
        return
    lexer = Lexer(file.read(), "test.cal")
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
                for res in result.value:
                    print(res)

if len(sys.argv) > 1:
    execute(sys.argv[1])