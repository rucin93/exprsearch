use expr_rust::{NumT, Operator, Expr};

pub const USE_JIT: bool = true;

pub const ANSWER: &[NumT] = &[1,1,2,3,5,8,13,21,34,55,89,144];

pub const INIT_X_MIN: NumT = -1;
pub const INIT_X_MAX: NumT = 1;
pub const INIT_Y_MIN: NumT = -1;
pub const INIT_Y_MAX: NumT = 1;

pub const MAX_LENGTH: usize = 10;
pub const MAX_CACHE_LENGTH: usize = 7;
pub const USE_MULTITHREAD: bool = true;
pub const LITERALS: &[NumT] = &[1, 2, 3];
pub const USE_PARENS: bool = true;
pub const PRUNE_CONST_EXPR: bool = true; // Skip constant-only expressions since we can easily find them

pub struct Matcher {}

impl Matcher {
    pub fn new() -> Self {
        Self {}
    }

    #[inline]
    pub fn match_one(&mut self, index: usize, output: NumT) -> bool {
        output == ANSWER[index]
    }

    pub fn match_final(self, _e_x: &Expr, _e_y: &Expr) -> bool {
        true
    }
}

impl Default for Matcher {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy)]
pub struct BinaryOp {
    pub op: Operator,
    pub len: usize,
}

#[derive(Clone, Copy)]
pub struct UnaryOp {
    pub op: Operator,
}

#[derive(Clone, Copy)]
pub struct AssignOp {
    pub op: Operator,
    pub len: usize,
}

#[derive(Clone, Copy)]
pub struct IncDecOp {
    pub op: Operator,
}

#[rustfmt::skip]
pub const BINARY_OPERATORS: &[BinaryOp] = &[
    BinaryOp { op: Operator::BitOr, len: 1 },
    BinaryOp { op: Operator::BitXor, len: 1 },
    BinaryOp { op: Operator::BitAnd, len: 1 },
    BinaryOp { op: Operator::Lt, len: 1 },
    BinaryOp { op: Operator::Gt, len: 1 },
    BinaryOp { op: Operator::Add, len: 1 },
    BinaryOp { op: Operator::Sub, len: 1 },
    BinaryOp { op: Operator::Mul, len: 1 },
    BinaryOp { op: Operator::Div, len: 1 },
    BinaryOp { op: Operator::Mod, len: 1 },
    BinaryOp { op: Operator::Eq, len: 2 },
    BinaryOp { op: Operator::Neq, len: 2 },
    BinaryOp { op: Operator::Leq, len: 2 },
    BinaryOp { op: Operator::Geq, len: 2 },
    BinaryOp { op: Operator::BitShl, len: 2 },
    BinaryOp { op: Operator::BitShr, len: 2 },
    // BinaryOp { op: Operator::Pow, len: 2 },
    BinaryOp { op: Operator::Or, len: 2 },
    BinaryOp { op: Operator::And, len: 2 },
];

#[rustfmt::skip]
pub const UNARY_OPERATORS: &[UnaryOp] = &[
    UnaryOp { op: Operator::Neg },
    UnaryOp { op: Operator::BitNot },
    UnaryOp { op: Operator::Not },
];

#[rustfmt::skip]
pub const ASSIGN_OPERATORS: &[AssignOp] = &[
    AssignOp { op: Operator::AssignEq, len: 1 },
    AssignOp { op: Operator::AddEq, len: 2 },
    AssignOp { op: Operator::SubEq, len: 2 },
    AssignOp { op: Operator::MulEq, len: 2 },
    AssignOp { op: Operator::DivEq, len: 2 },
    AssignOp { op: Operator::ModEq, len: 2 },
    AssignOp { op: Operator::BitOrEq, len: 2 },
    AssignOp { op: Operator::BitXorEq, len: 2 },
    AssignOp { op: Operator::BitAndEq, len: 2 },
    AssignOp { op: Operator::BitShlEq, len: 3 },
    AssignOp { op: Operator::BitShrEq, len: 3 },
];

#[rustfmt::skip]
pub const INCDEC_OPERATORS: &[IncDecOp] = &[
    IncDecOp { op: Operator::PreInc },
    IncDecOp { op: Operator::PreDec },
    IncDecOp { op: Operator::PostInc },
    IncDecOp { op: Operator::PostDec },
];

pub fn binary_ops_by_len(len: usize) -> impl Iterator<Item = &'static BinaryOp> {
    BINARY_OPERATORS.iter().filter(move |o| o.len == len)
}
