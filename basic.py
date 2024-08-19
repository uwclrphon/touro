from string_with_arrows import string_with_arrows
TT_INT = 'INT'
TT_FLOAT = 'FLOAT'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_POW = 'POW'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_EE = 'EE'
TT_NE = 'NE'
TT_LT = 'LT'
TT_LTE = 'LTE'
TT_GTE = 'GTE'
TT_EOF = 'EOF'
TT_IDENTIFIER = 'IDENTIFIER'
TT_KEYWORD = 'KEYWORD'
TT_EQ = 'EQ'
TT_GT = 'GT'
TT_PASS = 'PASS'
TT_NOPRINT = 'NOPRINT'
TT_IDNO = 'IDNO'
TT_LBRACE = 'LBRACE'
TT_RBRACE = 'RBRACE'
TT_COLON = 'COLON'
TT_COMMA = 'COMMA'
DIGITS = '0123456789'
LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
OP = [
    TT_PASS,
    TT_MINUS,
    TT_MUL,
    TT_DIV,
    TT_POW
]
LETTERS_DIGITS = LETTERS + DIGITS
KEYWORDS = [
    'int',
    'float',
    'and',
    'or',
    'not',
    'if',
    'elif',
    'else',
    'for',
    'while',
    'fun'
]
GT_TYPE_KEYWORDS = [
    'int',
    'float',
]
class Error:
    def __init__(self,pos_start,pos_end,error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details
    def as_string(self):
        result = f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}\n'
        result += f'{string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)}\n'
        result += f'    {self.error_name}:{self.details}'
        return result
class IllegalCharError(Error):
    def __init__(self,pos_start,pos_end,details,name='Illegal Character'):
        super().__init__(pos_start,pos_end,name, details)
class InvalidSyntaxError(Error):
    def __init__(self,pos_start,pos_end,details='',name='Invalid Syntax'):
        super().__init__(pos_start,pos_end,name, details)
class RTError(Error):
    def __init__(self,pos_start,pos_end,details,context,name='Runtime Error'):
        super().__init__(pos_start,pos_end,name, details)
        self.context = context
    def as_string(self):
        result = self.generate_traceback()
        result += f'{string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)}\n'
        result += f'    {self.error_name}:{self.details}'
        return result
    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        cts = self.context
        while cts:
            result = f'    File {pos.fn}, line {str(pos.ln + 1)}, in {cts.display_name}\n' + result
            pos = cts.parent_entry_pos
            cts = cts.parent
        return 'Traceback (most recent call last):\n' + result
class ZeroDivisionError(RTError):
    def __init__(self,pos_start,pos_end,details,context):
        super().__init__(pos_start,pos_end,details,context,'ZeroDivisionError')
class StartFunError(Error):
    def __init__(self,pos_start,pos_end,details):
        super().__init__(pos_start,pos_end,'StartFunError',details)
class VariableNotFoundError(RTError):
    def __init__(self,pos_start,pos_end,details,context):
        super().__init__(pos_start,pos_end,details,context,'VariableNotFoundError')
class ClassNotRightTypeError(Error):
    def __init__(self,pos_start,pos_end,details):
        super().__init__(pos_start,pos_end,'ClassNotRightTypeError',details)
class VariableDefinedError(RTError):
    def __init__(self,pos_start,pos_end,details,context):
        super().__init__(pos_start,pos_end,details,context,'VariableDefinedError')
class Types:
    def __init__(self,type_):
        self.type = type_
    def __str__(self):
        return self.type
class NoPrint(Types):
    def __init__(self):
        super().__init__('NoPrint')
class TypesBox:
    def __init__(self):
        self.types = []
    def set(self, type_):
        self.types.append(type_)
    def iftype(self,type_):
        for i in self.types:
            if i.type == type_:
                return True
        return False
    def __str__(self):
        return str(self.types)
class Position:
    def __init__(self, idx, ln, col,fn,ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt
    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1
        if current_char == '\n':
            self.ln += 1
            self.col = 0
        return self
    def back(self, current_char=None):
        self.idx -= 1
        self.col -= 1
        if current_char == '\n':
            self.ln -= 1
            self.col = 0
        return self
    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)
class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value
        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()
        if pos_end:
            self.pos_end = pos_end.copy()
    def __repr__(self):
        if self.value:return f'{self.type}:{self.value}'
        return f'{self.type}'
    def matches(self, type_, value):
        return self.type == type_ and self.value == value
class Lexer:
    def __init__(self,fn,text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1,fn,text)
        self.current_char = None
        self.advance()
    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None
    def back(self):
        self.pos.back(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None
    def make_tokens(self):
        tokens = []
        while self.current_char != None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS,pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS,pos_start=self.pos))
                self.advance()
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL,pos_start=self.pos))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV,pos_start=self.pos))
                self.advance()
            elif self.current_char == '^':
                tokens.append(Token(TT_POW,pos_start=self.pos))
                self.advance()
            elif self.current_char == ';':
                tokens.append(Token(TT_COLON,pos_start=self.pos))
                self.advance()
            elif self.current_char == '=':
                tok, error = self.make_equals()
                if error: return [], error
                tokens.append(tok)
            elif self.current_char == '!':
                tok, error = self.make_not_equals()
                if error: return [], error
                tokens.append(tok)
            elif self.current_char == ',':
                tokens.append(Token(TT_COMMA,pos_start=self.pos))
                self.advance()
            elif self.current_char == '<':
                tokens.append(self.make_less_than())
            elif self.current_char == '>':
                tokens.append(self.make_greater_than())
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN,pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN,pos_start=self.pos))
                self.advance()
            elif self.current_char == '{':
                tokens.append(Token(TT_LBRACE,pos_start=self.pos))
                self.advance()
            elif self.current_char == '}':
                tokens.append(Token(TT_RBRACE,pos_start=self.pos))
                self.advance()
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start,self.pos,f"'{char}'")
        tokens.append(Token(TT_EOF,pos_start=self.pos))
        return tokens,None
    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()
        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1:break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()
        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)
    def make_identifier(self):
        id_str = ''
        pos_start = self.pos.copy()
        while self.current_char != None and self.current_char in LETTERS + '_':
            id_str += self.current_char
            self.advance()
        tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
        return Token(tok_type, id_str, pos_start, self.pos)
    def make_not_equals(self):
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            return Token(TT_NE,pos_start=pos_start, pos_end=self.pos), None
        self.advance()
        return None, IllegalCharError(pos_start, self.pos, "'=' (after '!')")
    def make_equals(self):
        tok_type = TT_EQ
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            tok_type = TT_EE
        return Token(tok_type, pos_start=pos_start, pos_end=self.pos), None
    def make_less_than(self):
        tok_type = TT_LT
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            tok_type = TT_LTE
        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)
    def make_greater_than(self):
        tok_type = TT_GT
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            tok_type = TT_GTE
        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)
class NumberNode:
    def __init__(self,tok):
        self.tok = tok
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end
    def __repr__(self):
        return f'{self.tok}'
class BinOpNode:
    def __init__(self,left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node
        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end
    def __repr__(self):
        return f'({self.left_node}, {self.op_tok}, {self.right_node})'
class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node
        self.pos_start = self.op_tok.pos_start
        self.pos_end = self.node.pos_end
    def __repr__(self):
        return f'({self.op_tok}, {self.node})'
class VarAccessNode:
    def __init__(self,var_name_tok):
        self.var_name_tok = var_name_tok
        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end
    def __repr__(self):
        return f'{self.var_name_tok}'
class VarAssignNode:
    def __init__(self,var_name_tok, value_node,types=None):
        self.var_name_tok = var_name_tok
        self.value_node = value_node
        self.types = types
        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.value_node.pos_end
    def __repr__(self):
        return f'{self.var_name_tok} = {self.value_node}'
class VarReviseNode:
    def __init__(self,var_name_tok, value_node,types=None):
        self.var_name_tok = var_name_tok
        self.value_node = value_node
        self.types = types
        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.value_node.pos_end
    def __repr__(self):
        return f'{self.var_name_tok} = {self.value_node}'
class IfNode:
    def __init__(self,cases,else_case):
        self.cases = cases
        self.else_case = else_case
        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = (self.else_case or self.cases[len(self.cases)-1][1]).pos_end
    def __repr__(self):
        return f'If({self.cases},{self.else_case})'
class ForNode:
    def __init__(self,var_name_tok, start_value_node, end_value_node, step_value_node, body_node,types=None):
        self.var_name_tok = var_name_tok
        self.start_value_node = start_value_node
        self.end_value_node = end_value_node
        self.step_value_node = step_value_node
        self.body_node = body_node
        self.types = types
        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.body_node.pos_end
    def __repr__(self):
        return f'For({self.var_name_tok},{self.start_value_node},{self.end_value_node},{self.step_value_node},{self.body_node})'
class WhileNode:
    def __init__(self,condition_node,body_node):
        self.condition_node = condition_node
        self.body_node = body_node
        self.pos_start = self.condition_node.pos_start
        self.pos_end = self.body_node.pos_end
class TypeNode:
    def __init__(self,pos,type_):
        self.type = type_
        self.pos_start = pos.pos_start
        self.pos_end = pos.pos_end
class FuncDefNode:
    def __init__(self,var_name_tok, arg_name_toks, body_node):
        self.var_name_tok = var_name_tok
        self.arg_name_toks = arg_name_toks
        self.body_node = body_node
        if self.arg_name_toks:
            self.pos_start = self.arg_name_toks[0].pos_start
        elif self.var_name_tok:
            self.pos_start = self.var_name_tok.pos_start
        else:
            self.pos_start = self.body_node.pos_start
        self.pos_end = self.body_node.pos_end
    def __repr__(self):
        return f'FunDef({self.var_name_tok},{self.arg_name_toks},{self.body_node})'
class CallNode:
    def __init__(self,node_to_call, arg_nodes):
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes
        self.pos_start = self.node_to_call.pos_start
        if len(self.arg_nodes) > 0:
            self.pos_end = self.arg_nodes[len(self.arg_nodes)-1].pos_end
        else:
            self.pos_end = self.node_to_call.pos_end
class ParserResult:
    def __init__(self):
        self.error = None
        self.node = None
    def register(self,res):
        if isinstance(res,ParserResult):
            if res.error: self.error = res.error
            return res.node
        return res
    def success(self,node):
        self.node = node
        return self
    def failure(self,error):
        self.error = error
        return self
class Parser:
    def __init__(self,tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.current_tok = Token(TT_PASS,'')
        self.back = None
        self.advance()
    def advance(self):
        self.tok_idx += 1
        self.back = self.current_tok
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self
    def backs(self):
        self.tok_idx -= 1
        if self.tok_idx >= 0:
            self.current_tok = self.tokens[self.tok_idx]
        return self
    def parse(self):
        res = self.expr()
        if not res.error and self.current_tok.type != TT_EOF:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting EOF.'
            ))
        return res
    def if_expr(self):
        res = ParserResult()
        cases = []
        else_case = None
        if not self.current_tok.matches(TT_KEYWORD,'if'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting "if".'
            ))
        res.register(self.advance())
        condition = res.register(self.expr())
        if res.error: return res
        if not self.current_tok.type == TT_LBRACE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting "(".'
            ))
        res.register(self.advance())
        expr = res.register(self.expr())
        if res.error: return res
        if not self.current_tok.type == TT_RBRACE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting ")".'
            ))
        cases.append((condition,expr))
        res.register(self.advance())
        while self.current_tok.matches(TT_KEYWORD,'elif'):
            res.register(self.advance())
            condition = res.register(self.expr())
            if res.error: return res
            if not self.current_tok.type == TT_LBRACE:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start,self.current_tok.pos_end,'Expecting "(".'
                ))
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error: return res
            if not self.current_tok.type == TT_RBRACE:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start,self.current_tok.pos_end,'Expecting ")".'
                ))
            cases.append((condition,expr))
            res.register(self.advance())
        if self.current_tok.matches(TT_KEYWORD,'else'):
            res.register(self.advance())
            if not self.current_tok.type == TT_LBRACE:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start,self.current_tok.pos_end,'Expecting "(".'
                ))
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error: return res
            if not self.current_tok.type == TT_RBRACE:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start,self.current_tok.pos_end,'Expecting ")".'
            ))
            else_case = expr
            res.register(self.advance())
        return res.success(IfNode(cases,else_case))
    def for_expr(self):
        res = ParserResult()
        if not self.current_tok.matches(TT_KEYWORD,'for'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting "for".'
            ))
        res.register(self.advance())
        types = None
        if self.current_tok.type == TT_KEYWORD and self.current_tok.value in GT_TYPE_KEYWORDS:
            types = self.current_tok.value
            res.register(self.advance())
        if self.current_tok.type != TT_GT:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting "for".'
            ))
        res.register(self.advance())
        if self.current_tok.type != TT_IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting identifier.'
            ))
        var_name = self.current_tok
        res.register(self.advance())
        op = TypeNode(self.current_tok,self.current_tok.type)
        res.register(self.advance())
        if self.current_tok.type != TT_INT:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting integer.'
            ))
        step = res.register(self.expr())
        if self.current_tok.type != TT_LPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting "(".'
            ))
        res.register(self.advance())
        start_var = res.register(self.expr())
        if res.error: return res
        if self.current_tok.type != TT_COLON:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting ";".'
            ))
        res.register(self.advance())
        end_var = res.register(self.expr())
        if res.error: return res
        if self.current_tok.type != TT_RPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting ")".'
            ))
        if res.error: return res
        res.register(self.advance())
        if self.current_tok.type != TT_LBRACE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting "{".'
            ))
        res.register(self.advance())
        body = res.register(self.expr())
        if res.error: return res
        if self.current_tok.type != TT_RBRACE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting "}".'
            ))
        res.register(self.advance())
        return res.success(ForNode(var_name,start_var,end_var,step,body,op))
    def while_expr(self):
        res = ParserResult()
        if not self.current_tok.matches(TT_KEYWORD,'while'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting "while".'
            ))
        res.register(self.advance())
        condition = res.register(self.expr())
        if res.error: return res
        if self.current_tok.type != TT_LBRACE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting "{".'
            ))
        res.register(self.advance())
        body = res.register(self.expr())
        if res.error: return res
        if self.current_tok.type != TT_RBRACE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting "}".'
            ))
        res.register(self.advance())
        return res.success(WhileNode(condition,body))
    def fun_def(self):
        res = ParserResult()
        if not (self.current_tok.matches(TT_KEYWORD,'fun') or self.current_tok.type == TT_LT):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting "fun".'
            ))
        res.register(self.advance())
        arg_name_tok = []
        if self.current_tok.type == TT_IDENTIFIER:
            arg_name_tok.append(self.current_tok)
            res.register(self.advance())
            while self.current_tok.type == TT_COMMA:
                res.register(self.advance())
                if self.current_tok.type != TT_IDENTIFIER:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start,self.current_tok.pos_end,'Expecting identifier.'
                    ))
                arg_name_tok.append(self.current_tok)
                res.register(self.advance())
        if self.current_tok.type in GT_TYPE_KEYWORDS:
            res.register(self.advance())
        if self.current_tok.type != TT_GT:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting ">".'
            ))
        res.register(self.advance())
        if self.current_tok.type == TT_IDENTIFIER:
            var_name_tok = self.current_tok
            res.register(self.advance())
        else:
            var_name_tok = None
        if self.current_tok.type != TT_LPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting "(".'
            ))
        res.register(self.advance())
        if self.current_tok.type != TT_RPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting closing parenthesis.'
            ))
        res.register(self.advance())
        if self.current_tok.type != TT_LBRACE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting "{".'
            ))
        res.register(self.advance())
        node_to_return = res.register(self.expr())
        if res.error: return res
        if self.current_tok.type != TT_RBRACE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting "}".'
            ))
        res.register(self.advance())
        return res.success(FuncDefNode(var_name_tok,arg_name_tok,node_to_return))
    def atom(self):
        res = ParserResult()
        tok = self.current_tok
        if tok.type in (TT_INT,TT_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))
        elif tok.type == TT_IDENTIFIER:
            res.register(self.advance())
            return res.success(VarAccessNode(tok))
        elif tok.type == TT_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_tok.type == TT_RPAREN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(      
                    self.current_tok.pos_start,self.current_tok.pos_end,'Expecting closing parenthesis.'
                ))
        elif tok.matches(TT_KEYWORD,'if'):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)
        elif tok.matches(TT_KEYWORD,'for'):
            for_expr = res.register(self.for_expr())
            if res.error: return res
            return res.success(for_expr)
        elif tok.matches(TT_KEYWORD,'while'):
            while_expr = res.register(self.while_expr())
            if res.error: return res
            return res.success(while_expr)
        elif tok.matches(TT_KEYWORD,'fun'):
            func_def = res.register(self.fun_def())
            if res.error: return res
            return func_def
        return res.failure(InvalidSyntaxError(
            tok.pos_start,tok.pos_end,'Expecting integer or float.'
        ))
    def power(self):
        return self.bin_op(self.call,(TT_POW,),self.factor)
    def call(self):
        res = ParserResult()
        node = res.register(self.atom())
        if res.error: return res
        if self.current_tok.type == TT_LPAREN:
            res.register(self.advance())
            arg_nodes = []
            if self.current_tok.type == TT_RPAREN:
                res.register(self.advance())
            else:
                arg_nodes.append(res.register(self.expr()))
                if res.error:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start,self.current_tok.pos_end,'Expecting closing parenthesis.'
                    ))
                while self.current_tok.type == TT_COMMA:
                    res.register(self.advance())
                    arg_nodes.append(res.register(self.expr()))
                    if res.error:
                        return res.failure(InvalidSyntaxError(
                            self.current_tok.pos_start,self.current_tok.pos_end,'Expecting closing parenthesis.'
                        ))
                if self.current_tok.type != TT_RPAREN:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start,self.current_tok.pos_end,'Expecting closing parenthesis.'
                    ))
                res.register(self.advance())
            return res.success(CallNode(node,arg_nodes))
        return res.success(node)
                    
    def factor(self):
        res = ParserResult()
        tok = self.current_tok
        if tok.type in (TT_PLUS,TT_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok,factor))
        return self.power()
    def term(self):
        return self.bin_op(self.factor,(TT_MUL,TT_DIV))
    def arith_expr(self):
        return self.bin_op(self.term,(TT_PLUS,TT_MINUS))
    def comp_expr(self):
        res = ParserResult()
        if self.current_tok.matches(TT_KEYWORD,'not'):
            op_tok = self.current_tok
            res.register(self.advance())
            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryOpNode(op_tok,node))
        node = res.register(self.bin_op(self.arith_expr,(TT_EE,TT_NE,TT_LT,TT_GT,TT_LTE,TT_GTE)))
        if res.error: return res
        return res.success(node)
    def comper(self,func,types):
        res = ParserResult()
        if self.current_tok.type != TT_IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting identifier.'
            ))
        var_name = self.current_tok
        res.register(self.advance())
        if self.current_tok.type != TT_EQ:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting equals.'
            ))
        res.register(self.advance())
        expr = res.register(self.expr())
        if res.error: return res
        return res.success(func(var_name,expr))
    def expr(self):
        res = ParserResult()
        if self.current_tok.type == TT_KEYWORD and self.current_tok.value in GT_TYPE_KEYWORDS:
            types = self.current_tok.value
            res.register(self.advance())
            if self.current_tok.type == TT_GT:  
                res.register(self.advance())
                return self.comper(VarAssignNode,types)
            if self.current_tok.type == TT_LT:
                return self.fun_def()
            return res.failure(ClassNotRightTypeError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting key.'
            ))
        if self.current_tok.type == TT_GT:
            res.register(self.advance())
            return self.comper(VarAssignNode,None)
        if self.current_tok.type == TT_IDENTIFIER:
            cos = TT_PASS
            var_name = self.current_tok
            res.register(self.advance())
            if self.current_tok.type != TT_EQ:
                cos = TT_IDNO
                res.register(self.backs())
            if cos == TT_PASS:
                res.register(self.advance())
                expr = res.register(self.expr())
                if res.error: return res
                return res.success(VarReviseNode(var_name,expr))
        if self.current_tok.matches(TT_KEYWORD,'fun') and self.tok_idx == 0:
            return res.failure(StartFunError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting "fun" at the start of the program.'
            ))
        if self.current_tok.type == TT_LT:
            return self.fun_def()
        node = self.bin_op(self.comp_expr,((TT_KEYWORD,'and'),(TT_KEYWORD,'or')))
        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,'Expecting integer or float.'
            ))
        return node
    def bin_op(self,func,ops, factor=None):
        if factor is None: factor = func
        res = ParserResult()
        left = res.register(func())
        if res.error: return res
        while self.current_tok.type in ops or (self.current_tok.type,self.current_tok.value) in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(factor())
            if res.error: return res
            left = BinOpNode(left, op_tok, right)
        return res.success(left)
class RTResult:
    def __init__(self,value=None, error=None):
        self.value = value
        self.error = error
    def register(self,res):
        if res.error: self.error = res.error
        return res.value
    def success(self,value):
        self.value = value
        return self
    def failure(self,error):
        self.error = error
        return self
class Value:
    def __init__(self) -> None:
        self.set_pos()
        self.set_context()
    def set_pos(self,pos_start=None,pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    def set_context(self,context=None):
        self.context = context
        return self
    def added_to(self,other):return None,self.execute()
    def subeed_by(self,other):return None,self.execute()
    def mulled_by(self,other):return None,self.execute()
    def divded_by(self,other):return None,self.execute()
    def anded_by(self,other):return None,self.execute()
    def orred_by(self,other):return None,self.execute()
    def notted(self):return None,self.execute()
    def if_true(self):return None,self.execute()
    def powed_by(self,other):return None,self.execute()
    def get_comparison_eq(self,other):return None,self.execute()
    def get_comparison_ne(self,other):return None,self.execute()
    def get_comparison_lt(self,other):return None,self.execute()
    def get_comparison_lte(self,other):return None,self.execute()
    def get_comparison_gt(self,other):return None,self.execute()
    def get_comparison_gte(self,other):return None,self.execute()
    def execute(self):
        return RTError(
            self.pos_start,self.pos_end,'Can not execute this.',self.context
        )
class Number(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def added_to(self,other):
        if isinstance(other,Number):
            return Number(self.value + other.value).set_context(self.context).set_pos(self.pos_start,other.pos_end),None
    def subeed_by(self,other):
        if isinstance(other,Number):
            return Number(self.value - other.value).set_context(self.context).set_pos(self.pos_start,other.pos_end),None
    def mulled_by(self,other):
        if isinstance(other,Number):
            return Number(self.value * other.value).set_context(self.context).set_pos(self.pos_start,other.pos_end),None
    def divded_by(self,other):
        if isinstance(other,Number):
            if other.value == 0:
                return None,ZeroDivisionError(
                    self.pos_start,self.pos_end,'Division by zero.',self.context

                )
            return Number(self.value / other.value).set_context(self.context).set_pos(self.pos_start,other.pos_end),None
    def powed_by(self,other):
        if isinstance(other,Number):
            return Number(self.value ** other.value).set_context(self.context).set_pos(self.pos_start,other.pos_end),None
    def get_comparison_eq(self,other):
        if isinstance(other,Number):
            return Number(int(self.value == other.value)).set_context(self.context).set_pos(self.pos_start,other.pos_end),None
    def get_comparison_ne(self,other):
        if isinstance(other,Number):
            return Number(int(self.value != other.value)).set_context(self.context).set_pos(self.pos_start,other.pos_end),None
    def get_comparison_lt(self,other):
        if isinstance(other,Number):
            return Number(int(self.value < other.value)).set_context(self.context).set_pos(self.pos_start,other.pos_end),None
    def get_comparison_gt(self,other):
        if isinstance(other,Number):
            return Number(int(self.value > other.value)).set_context(self.context).set_pos(self.pos_start,other.pos_end),None
    def get_comparison_lte(self,other):
        if isinstance(other,Number):
            return Number(int(self.value <= other.value)).set_context(self.context).set_pos(self.pos_start,other.pos_end),None
    def get_comparison_gte(self,other):
        if isinstance(other,Number):
            return Number(int(self.value >= other.value)).set_context(self.context).set_pos(self.pos_start,other.pos_end),None
    def anded_by(self,other):
        if isinstance(other,Number):
            return Number(int(self.value and other.value)).set_context(self.context).set_pos(self.pos_start,other.pos_end),None
    def ored_by(self,other):
        if isinstance(other,Number):
            return Number(int(self.value or other.value)).set_context(self.context).set_pos(self.pos_start,other.pos_end),None
    def notted(self):
        return Number(int(not self.value)).set_context(self.context).set_pos(self.pos_start,self.pos_end)
    def if_true(self):
        return self.value != 0
    def __repr__(self):
        return str(self.value)
class Function(Value):
    def __init__(self,name,body_node,arg_names):
        self.name = name or "<anonymous>"
        self.body_node = body_node
        self.arg_names = arg_names
        self.context = None
    def _execute(self,args):
        res = RTResult()
        interpreter = Interpreter()
        new_context = Context(self.name,self.context,self.pos_start)
        new_context.symbol_table = SymbolTable(self.context.symbol_table)
        if len(args) > len(self.arg_names):
            return res.failure(RTError(
                self.pos_start,self.pos_end,'Too many arguments passed.',self.context
            ))
        if len(args) < len(self.arg_names):
            return res.failure(RTError(
                self.pos_start,self.pos_end,'Too few arguments passed.',self.context
            ))
        for i in range(len(args)):
            arg_name = self.arg_names[i]
            value = args[i]
            value.set_context(new_context)
            new_context.symbol_table.set(arg_name,value)
        value = res.register(interpreter.visit(self.body_node,new_context))
        if res.error: return res
        return res.success(value)
    def copy(self):
        copy = Function(self.name,self.body_node,self.arg_names)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start,self.pos_end)
        return copy
    def __repr__(self):
        return f"<function {self.name}>"
class SymbolTable:
    def __init__(self,parent=None):
        self.symbols = {}
        self.parent = parent
    def get(self,name):
        value = self.symbols.get(name,None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value
    def set(self,name,value):
        self.symbols[name] = value
class Context:
    def __init__(self,display_name,parent=None,parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = SymbolTable()
        self.nop = TT_PASS
        self.types = TypesBox()
class Interpreter:
    def visit(self,node,context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self,method_name, self.no_visit_method)
        return method(node, context)
    def no_visit_method(self,node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')
    def visit_NumberNode(self,node, context):
        return RTResult().success(Number(node.tok.value).set_context(context).set_pos(node.pos_start,node.pos_end))
    def visit_VarAccessNode(self,node,context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = context.symbol_table.get(var_name)
        if not value:
            return res.failure(VariableNotFoundError(
                node.pos_start,node.pos_end,
                f"'{var_name}' is not defined.", context
            ))
        return res.success(value)
    def visit_VarAssignNode(self,node,context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = res.register(self.visit(node.value_node,context))
        if res.error: return res
        if context.symbol_table.get(var_name):
            return res.failure(VariableDefinedError(
                node.pos_start,node.pos_end,
                f"'{var_name}' is already defined.", context
            ))
        context.symbol_table.set(var_name,value)
        context.types.set(NoPrint())
        return res.success(value)
    def visit_VarReviseNode(self,node,context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = res.register(self.visit(node.value_node,context))
        if res.error: return res
        res.register(self.visit_VarAccessNode(node,context))
        if res.error: return res
        context.symbol_table.set(var_name,value)
        context.types.set(NoPrint())
        return res.success(value)
    def visit_BinOpNode(self,node,context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error: return res
        right = res.register(self.visit(node.right_node, context))
        if res.error: return res
        if node.op_tok.type == TT_PLUS:
            result,error = left.added_to(right)
        elif node.op_tok.type == TT_MINUS:
            result,error =  left.subeed_by(right)
        elif node.op_tok.type == TT_MUL:
            result,error =  left.mulled_by(right)
        elif node.op_tok.type == TT_DIV:
            result,error =  left.divded_by(right)
        elif node.op_tok.type == TT_POW:
            result,error =  left.powed_by(right)
        elif node.op_tok.type == TT_EE:
            result,error = left.get_comparison_eq(right)
        elif node.op_tok.type == TT_NE:
            result,error = left.get_comparison_ne(right)
        elif node.op_tok.type == TT_LT:
            result,error = left.get_comparison_lt(right)
        elif node.op_tok.type == TT_GT:
            result,error = left.get_comparison_gt(right)
        elif node.op_tok.type == TT_LTE:
            result,error = left.get_comparison_lte(right)
        elif node.op_tok.type == TT_GTE:
            result,error = left.get_comparison_gte(right)
        elif node.op_tok.matches(TT_KEYWORD, 'and'):
            result,error = left.anded_by(right)
        elif node.op_tok.matches(TT_KEYWORD, 'or'):
            result,error = left.ored_by(right)
        if error: return res.failure(error)
        else: return res.success(result.set_pos(node.pos_start,node.pos_end))
    def visit_UnaryOpNode(self,node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.error: return res
        error = None
        if node.op_tok.type == TT_MINUS:
            number,error = number.mulled_by(Number(-1))
        elif node.op_tok.matches(TT_KEYWORD, 'not'):
            number,error = number.notted()
        if error: return res.failure(error)
        return res.success(number.set_pos(node.pos_start,node.pos_end))
    def visit_IfNode(self,node, context):
        res = RTResult()
        for condition, expr in node.cases:
            condition_value = res.register(self.visit(condition, context))
            if res.error: return res
            if condition_value.if_true():
                expr_value = res.register(self.visit(expr, context))
                if res.error: return res
                return res.success(expr_value)
        if node.else_case:
            expr_value = res.register(self.visit(node.else_case, context))
            if res.error: return res
            return res.success(expr_value)
        context.types.set(NoPrint())
        return res.success(None)
    def visit_ForNode(self,node,context):
        res = RTResult()
        elements = []
        start_value = res.register(self.visit(node.start_value_node, context))
        if res.error: return res
        end_value = res.register(self.visit(node.end_value_node, context))
        step_value = res.register(self.visit(node.step_value_node, context))
        if res.error: return res
        types = res.register(self.visit(node.types, context))
        i = start_value.value
        if types in (TT_MINUS,TT_DIV):
            condition = lambda: i > end_value.value
        else:
            condition = lambda: i < end_value.value
        while condition():
            context.symbol_table.set(node.var_name_tok.value,Number(i))
            if types == TT_PLUS:
                i += step_value.value
            elif types == TT_MINUS:
                i -= step_value.value
            elif types == TT_DIV:
                i /= step_value.value
            else:
                i *= step_value.value
            res.register(self.visit(node.body_node, context))
            if res.error: return res
        context.types.set(NoPrint())
        return res.success(elements)
    def visit_WhileNode(self,node,context):
        res = RTResult()
        while True:
            condition = res.register(self.visit(node.condition_node, context))
            if res.error: return res
            if not condition.if_true(): break
            value = res.register(self.visit(node.body_node, context))
            if res.error: return res
        context.types.set(NoPrint())
        return res.success(value)
    def visit_TypeNode(self,node,context):
        res = RTResult()
        return res.success(node.type)
    def visit_FuncDefNode(self,node,context):
        res = RTResult()
        func_name =  node.var_name_tok.value if node.var_name_tok else None
        body = node.body_node
        arg_names = [arg_name.value for arg_name in node.arg_name_toks]
        func_value = Function(func_name,body,arg_names).set_context(context).set_pos(node.pos_start,node.pos_end)
        if node.var_name_tok:
            context.symbol_table.set(func_name,func_value)
        context.types.set(NoPrint())
        return res.success(func_value)
    def visit_CallNode(self,node,context):
        res = RTResult()
        args = []
        value_to_call = res.register(self.visit(node.node_to_call, context))
        if res.error: return res
        value_to_call = value_to_call.copy().set_pos(node.pos_start,node.pos_end)
        for arg_node in node.arg_nodes:
            args.append(res.register(self.visit(arg_node, context)))
            if res.error: return res
        return_value = res.register(value_to_call._execute(args))
        if res.error: return res
        return res.success(return_value)

global_symbol_table = SymbolTable()
global_symbol_table.set('null',Number(0))
global_symbol_table.set('True',Number(1))
global_symbol_table.set('False',Number(0))
def run(fn,text):
    context = Context('<program>')
    lexer = Lexer(fn,text)
    token, error = lexer.make_tokens()
    if error: return None, error,context
    parser = Parser(token)
    ast = parser.parse()
    if ast.error: return None, ast.error,context
    interpreter = Interpreter()
    context.symbol_table = global_symbol_table
    res = interpreter.visit(ast.node, context)
    return res.value,res.error,context