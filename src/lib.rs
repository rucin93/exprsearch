use std::i64;
use std::sync::Arc;
use std::hash::{Hash, Hasher};

pub mod jit;

pub type NumT = i64;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operator {
    // Assignment operators
    AssignEq = 0x00,
    BitOrEq = 0x01,
    BitXorEq = 0x02,
    BitAndEq = 0x03,
    BitShlEq = 0x04,
    BitShrEq = 0x05,
    AddEq = 0x06,
    SubEq = 0x07,
    MulEq = 0x08,
    DivEq = 0x09,
    ModEq = 0x0A,
    // Binary operators
    Or = 0x20,
    And = 0x30,
    BitOr = 0x40,
    BitXor = 0x50,
    BitAnd = 0x60,
    Eq = 0x70,
    Neq = 0x71,
    Lt = 0x80,
    Leq = 0x81,
    Gt = 0x82,
    Geq = 0x83,
    BitShl = 0x90,
    BitShr = 0x91,
    Add = 0xA0,
    Sub = 0xA1,
    Mul = 0xB0,
    Div = 0xB1,
    Mod = 0xB2,
    Pow = 0xB3,  // **
    // Unary operators
    Neg = 0xC0,
    BitNot = 0xC1,
    Not = 0xC2,
    // Pre-increment/decrement operators
    PreInc = 0xC3,
    PreDec = 0xC4,
    // Post-increment/decrement operators
    PostInc = 0xD0,
    PostDec = 0xD1,
    // Parentheses
    Parens = 0xE0,
    // Operands
    Var = 0xF0,
    VarY = 0xF1, // Added for second variable
    Literal = 0xFF,
}

impl Operator {
    pub fn from_u8(n: u8) -> Option<Operator> {
        match n {
            0x00 => Some(Operator::AssignEq),
            0x01 => Some(Operator::BitOrEq),
            0x02 => Some(Operator::BitXorEq),
            0x03 => Some(Operator::BitAndEq),
            0x04 => Some(Operator::BitShlEq),
            0x05 => Some(Operator::BitShrEq),
            0x06 => Some(Operator::AddEq),
            0x07 => Some(Operator::SubEq),
            0x08 => Some(Operator::MulEq),
            0x09 => Some(Operator::DivEq),
            0x0A => Some(Operator::ModEq),
            0x20 => Some(Operator::Or),
            0x30 => Some(Operator::And),
            0x40 => Some(Operator::BitOr),
            0x50 => Some(Operator::BitXor),
            0x60 => Some(Operator::BitAnd),
            0x70 => Some(Operator::Eq),
            0x71 => Some(Operator::Neq),
            0x80 => Some(Operator::Lt),
            0x81 => Some(Operator::Leq),
            0x82 => Some(Operator::Gt),
            0x83 => Some(Operator::Geq),
            0x90 => Some(Operator::BitShl),
            0x91 => Some(Operator::BitShr),
            0xA0 => Some(Operator::Add),
            0xA1 => Some(Operator::Sub),
            0xB0 => Some(Operator::Mul),
            0xB1 => Some(Operator::Div),
            0xB2 => Some(Operator::Mod),
            0xB3 => Some(Operator::Pow),
            0xC0 => Some(Operator::Neg),
            0xC1 => Some(Operator::BitNot),
            0xC2 => Some(Operator::Not),
            0xC3 => Some(Operator::PreInc),
            0xC4 => Some(Operator::PreDec),
            0xD0 => Some(Operator::PostInc),
            0xD1 => Some(Operator::PostDec),
            0xE0 => Some(Operator::Parens),
            0xF0 => Some(Operator::Var),
            0xF1 => Some(Operator::VarY),
            0xFF => Some(Operator::Literal),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct Expr {
    pub left: Option<Arc<Expr>>,
    pub right: Option<Arc<Expr>>,
    pub literal: NumT,
    pub op: Operator,
    pub jit: Option<Arc<jit::Jit>>,
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        // Semantic equivalence check
        let range = 4;
        
        // If JIT is available, use it
        if let (Some(jit1), Some(jit2)) = (&self.jit, &other.jit) {
            let f1 = jit1.func();
            let f2 = jit2.func();
            
            for x_val in -range..=range {
                for y_val in -range..=range {
                     let mut x = x_val;
                     let mut y = y_val;
                     // JIT signature: fn(*mut i64, *mut i64) -> i64
                     let r1 = unsafe { f1(&mut x, &mut y) };
                     let x1 = x; let y1 = y;

                     x = x_val; y = y_val;
                     let r2 = unsafe { f2(&mut x, &mut y) };
                     let x2 = x; let y2 = y;

                     if r1 != r2 || x1 != x2 || y1 != y2 {
                         return false;
                     }
                }
            }
            return true;
        }

        // Fallback to naive
        for x_val in -range..=range {
            for y_val in -range..=range {
                let mut x1 = x_val;
                let mut y1 = y_val;
                let mut f1 = false;
                let r1 = naive_eval(self, &mut x1, &mut y1, &mut f1);

                let mut x2 = x_val;
                let mut y2 = y_val;
                let mut f2 = false;
                let r2 = naive_eval(other, &mut x2, &mut y2, &mut f2);

                if r1 != r2 || x1 != x2 || y1 != y2 {
                    return false;
                }
            }
        }
        true
    }
}

impl Eq for Expr {}

impl Hash for Expr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let range = 4;
        
        if let Some(jit) = &self.jit {
            let f = jit.func();
            for x_val in -range..=range {
                for y_val in -range..=range {
                    let mut x = x_val;
                    let mut y = y_val;
                    let r = unsafe { f(&mut x, &mut y) };
                    
                    r.hash(state);
                    x.hash(state);
                    y.hash(state);
                }
            }
            return;
        }

        for x_val in -range..=range {
            for y_val in -range..=range {
                let mut x = x_val;
                let mut y = y_val;
                let mut fatal = false;
                let r = naive_eval(self, &mut x, &mut y, &mut fatal);
                
                r.hash(state);
                x.hash(state);
                y.hash(state);
            }
        }
    }
}

impl Expr {
    pub fn is_assignment(&self) -> bool {
        (self.op as u8) < 0x10
    }
}

fn print_node(e: &Expr, var_names: &[char]) {
    match e.op {
        Operator::Or => print!("||"),
        Operator::And => print!("&&"),
        Operator::BitOr => print!("|"),
        Operator::BitXor => print!("^"),
        Operator::BitAnd => print!("&"),
        Operator::Eq => print!("=="),
        Operator::Neq => print!("!="),
        Operator::Lt => print!("<"),
        Operator::Leq => print!("<="),
        Operator::Gt => print!(">"),
        Operator::Geq => print!(">="),
        Operator::BitShl => print!("<<"),
        Operator::BitShr => print!(">>"),
        Operator::Add => print!("+"),
        Operator::Sub => print!("-"),
        Operator::Mul => print!("*"),
        Operator::Div => print!("/"),
        Operator::Mod => print!("%"),
        Operator::Pow => print!("**"),
        Operator::Neg => print!("-"),
        Operator::BitNot => print!("~"),
        Operator::Not => print!("!"),
        Operator::PreInc => print!("++"),
        Operator::PreDec => print!("--"),
        Operator::PostInc | Operator::PostDec => {}
        Operator::Parens => print!("("),
        Operator::Literal => print!("{}", e.literal),
        Operator::AssignEq => print!("="),
        Operator::BitOrEq => print!("|="),
        Operator::BitXorEq => print!("^="),
        Operator::BitAndEq => print!("&="),
        Operator::BitShlEq => print!("<<="),
        Operator::BitShrEq => print!(">>="),
        Operator::AddEq => print!("+="),
        Operator::SubEq => print!("-="),
        Operator::MulEq => print!("*="),
        Operator::DivEq => print!("/="),
        Operator::ModEq => print!("%="),
        Operator::Var | Operator::VarY => {
            let idx = (e.op as usize) & 0xF;
            if idx < var_names.len() {
                print!("{}", var_names[idx]);
            }
        }
    }
}

/// Check if operator needs parentheses when used as child of parent_op
fn needs_parens(child_op: Operator, parent_op: Operator, is_right: bool) -> bool {
    // Assignment operators don't need parens around their operands
    match parent_op {
        Operator::AssignEq | Operator::AddEq | Operator::SubEq | 
        Operator::MulEq | Operator::DivEq | Operator::ModEq |
        Operator::BitOrEq | Operator::BitXorEq | Operator::BitAndEq |
        Operator::BitShlEq | Operator::BitShrEq => return false,
        _ => {}
    }
    
    // Get precedence levels (higher = binds tighter)
    let prec = |op: Operator| -> u8 {
        match op {
            Operator::Or => 1,
            Operator::And => 2,
            Operator::BitOr => 3,
            Operator::BitXor => 4,
            Operator::BitAnd => 5,
            Operator::Eq | Operator::Neq => 6,
            Operator::Lt | Operator::Leq | Operator::Gt | Operator::Geq => 7,
            Operator::BitShl | Operator::BitShr => 8,
            Operator::Add | Operator::Sub => 9,
            Operator::Mul | Operator::Div | Operator::Mod => 10,
            Operator::Pow => 11,  // Highest binary precedence
            _ => 100,  // Variables, literals, unary, assignment - don't need parens
        }
    };
    
    let child_prec = prec(child_op);
    let parent_prec = prec(parent_op);
    
    // Need parens if child has lower precedence than parent
    // For right-associative Pow, also need parens for same precedence on left
    if child_prec < parent_prec {
        return true;
    }
    if parent_op == Operator::Pow && !is_right && child_prec == parent_prec {
        return true;
    }
    false
}

pub fn print_expression(e: &Expr, var_names: &[char]) {
    print_expr_with_parent(e, var_names, None, false);
}

fn print_expr_with_parent(e: &Expr, var_names: &[char], parent_op: Option<Operator>, is_right: bool) {
    let wrap = parent_op.map(|p| needs_parens(e.op, p, is_right)).unwrap_or(false);
    
    if wrap {
        print!("(");
    }
    
    if let Some(ref left) = e.left {
        print_expr_with_parent(left, var_names, Some(e.op), false);
    }
    print_node(e, var_names);
    if let Some(ref right) = e.right {
        print_expr_with_parent(right, var_names, Some(e.op), true);
        if e.op == Operator::Parens {
            print!(")");
        }
    }
    if e.op == Operator::PostInc {
        print!("++");
    }
    if e.op == Operator::PostDec {
        print!("--");
    }
    
    if wrap {
        print!(")");
    }
}

pub fn naive_eval(e: &Expr, x: &mut NumT, y: &mut NumT, fatal: &mut bool) -> NumT {
    let mut l = 0;
    let mut r = 0;
    
    if let Some(ref left) = e.left {
        l = naive_eval(left, x, y, fatal);
    }
    if let Some(ref right) = e.right {
        r = naive_eval(right, x, y, fatal);
    }

    match e.op {
        Operator::AssignEq | Operator::BitOrEq | Operator::BitXorEq | Operator::BitAndEq |
        Operator::BitShlEq | Operator::BitShrEq | Operator::AddEq | Operator::SubEq |
        Operator::MulEq | Operator::DivEq | Operator::ModEq => {
            let target_is_x = if let Some(ref left) = e.left {
                left.op == Operator::Var
            } else {
                false
            };
            
            let target = if target_is_x { x } else { y };
            
            match e.op {
                Operator::AssignEq => { *target = r; *target }
                Operator::BitOrEq => { *target |= r; *target }
                Operator::BitXorEq => { *target ^= r; *target }
                Operator::BitAndEq => { *target &= r; *target }
                Operator::BitShlEq => { *target <<= r; *target }
                Operator::BitShrEq => { *target >>= r; *target }
                Operator::AddEq => { *target += r; *target }
                Operator::SubEq => { *target -= r; *target }
                Operator::MulEq => { *target *= r; *target }
                Operator::DivEq => {
                    if r == 0 || (*target == i64::MIN && r == -1) {
                        *fatal = true;
                        0 
                    } else {
                        *target /= r;
                        *target
                    }
                }
                Operator::ModEq => {
                    if r == 0 || (*target == i64::MIN && r == -1) {
                        *fatal = true;
                        0
                    } else {
                        *target %= r;
                        *target
                    }
                }
                _ => unreachable!(),
            }
        }
        Operator::Or => if l != 0 || r != 0 { 1 } else { 0 },
        Operator::And => if l != 0 && r != 0 { 1 } else { 0 },
        Operator::BitOr => l | r,
        Operator::BitXor => l ^ r,
        Operator::BitAnd => l & r,
        Operator::Eq => if l == r { 1 } else { 0 },
        Operator::Neq => if l != r { 1 } else { 0 },
        Operator::Lt => if l < r { 1 } else { 0 },
        Operator::Leq => if l <= r { 1 } else { 0 },
        Operator::Gt => if l > r { 1 } else { 0 },
        Operator::Geq => if l >= r { 1 } else { 0 },
        Operator::BitShl => l << r,
        Operator::BitShr => l >> r,
        Operator::Add => l.wrapping_add(r),
        Operator::Sub => l.wrapping_sub(r),
        Operator::Mul => l.wrapping_mul(r),
        Operator::Div => {
            if r == 0 || (l == i64::MIN && r == -1) {
                *fatal = true;
                0
            } else {
                l / r
            }
        }
        Operator::Mod => {
            if r == 0 || (l == i64::MIN && r == -1) {
                *fatal = true;
                0
            } else {
                l % r
            }
        }
        Operator::Pow => {
            // Handle negative exponents and overflow
            if r < 0 {
                if l == 0 {
                    *fatal = true;
                    0
                } else if l == 1 {
                    1
                } else if l == -1 {
                    if r % 2 == 0 { 1 } else { -1 }
                } else {
                    0  // |l| > 1 with negative exponent -> 0 in integer math
                }
            } else if let Ok(exp) = u32::try_from(r) {
                l.checked_pow(exp).unwrap_or_else(|| {
                    *fatal = true;
                    0
                })
            } else {
                *fatal = true;  // Exponent too large
                0
            }
        }
        Operator::Neg => r.wrapping_neg(),
        Operator::BitNot => !r,
        Operator::Not => if r == 0 { 1 } else { 0 },
        Operator::PreInc => {
             let target_is_x = if let Some(ref right) = e.right {
                right.op == Operator::Var
             } else { false };
             let target = if target_is_x { x } else { y };
             *target = target.wrapping_add(1);
             *target
        }
        Operator::PreDec => {
             let target_is_x = if let Some(ref right) = e.right {
                right.op == Operator::Var
             } else { false };
             let target = if target_is_x { x } else { y };
             *target = target.wrapping_sub(1);
             *target
        }
        Operator::PostInc => {
             let target_is_x = if let Some(ref right) = e.right {
                right.op == Operator::Var
             } else { false };
             let target = if target_is_x { x } else { y };
             let val = *target;
             *target = target.wrapping_add(1);
             val
        }
        Operator::PostDec => {
             let target_is_x = if let Some(ref right) = e.right {
                right.op == Operator::Var
             } else { false };
             let target = if target_is_x { x } else { y };
             let val = *target;
             *target = target.wrapping_sub(1);
             val
        }
        Operator::Parens => r,
        Operator::Literal => e.literal,
        Operator::Var => *x,
        Operator::VarY => *y,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create a literal expression
    fn lit(val: NumT) -> Arc<Expr> {
        Arc::new(Expr {
            left: None,
            right: None,
            literal: val,
            op: Operator::Literal,
            jit: None,
        })
    }

    // Helper function to create a variable x expression
    fn var_x() -> Arc<Expr> {
        Arc::new(Expr {
            left: None,
            right: None,
            literal: 0,
            op: Operator::Var,
            jit: None,
        })
    }

    // Helper function to create a variable y expression
    fn var_y() -> Arc<Expr> {
        Arc::new(Expr {
            left: None,
            right: None,
            literal: 0,
            op: Operator::VarY,
            jit: None,
        })
    }

    // Helper function to create a binary expression
    fn binary(left: Arc<Expr>, op: Operator, right: Arc<Expr>) -> Expr {
        Expr {
            left: Some(left),
            right: Some(right),
            literal: 0,
            op,
            jit: None,
        }
    }

    // Helper function to create a unary expression
    fn unary(op: Operator, right: Arc<Expr>) -> Expr {
        Expr {
            left: None,
            right: Some(right),
            literal: 0,
            op,
            jit: None,
        }
    }

    // ==================== Operator::from_u8 Tests ====================

    #[test]
    fn test_operator_from_u8_valid() {
        assert_eq!(Operator::from_u8(0x00), Some(Operator::AssignEq));
        assert_eq!(Operator::from_u8(0x20), Some(Operator::Or));
        assert_eq!(Operator::from_u8(0x30), Some(Operator::And));
        assert_eq!(Operator::from_u8(0xA0), Some(Operator::Add));
        assert_eq!(Operator::from_u8(0xA1), Some(Operator::Sub));
        assert_eq!(Operator::from_u8(0xB0), Some(Operator::Mul));
        assert_eq!(Operator::from_u8(0xB1), Some(Operator::Div));
        assert_eq!(Operator::from_u8(0xB2), Some(Operator::Mod));
        assert_eq!(Operator::from_u8(0xB3), Some(Operator::Pow));
        assert_eq!(Operator::from_u8(0xC0), Some(Operator::Neg));
        assert_eq!(Operator::from_u8(0xF0), Some(Operator::Var));
        assert_eq!(Operator::from_u8(0xF1), Some(Operator::VarY));
        assert_eq!(Operator::from_u8(0xFF), Some(Operator::Literal));
    }

    #[test]
    fn test_operator_from_u8_invalid() {
        assert_eq!(Operator::from_u8(0x10), None);
        assert_eq!(Operator::from_u8(0x15), None);
        assert_eq!(Operator::from_u8(0xFE), None);
    }

    // ==================== Arithmetic Operations Tests ====================

    #[test]
    fn test_eval_add() {
        let e = binary(lit(10), Operator::Add, lit(20));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 30);
        assert!(!fatal);
    }

    #[test]
    fn test_eval_sub() {
        let e = binary(lit(30), Operator::Sub, lit(10));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 20);
        assert!(!fatal);
    }

    #[test]
    fn test_eval_mul() {
        let e = binary(lit(6), Operator::Mul, lit(7));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 42);
        assert!(!fatal);
    }

    #[test]
    fn test_eval_div() {
        let e = binary(lit(42), Operator::Div, lit(6));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 7);
        assert!(!fatal);
    }

    #[test]
    fn test_eval_div_by_zero() {
        let e = binary(lit(10), Operator::Div, lit(0));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        let result = naive_eval(&e, &mut x, &mut y, &mut fatal);
        assert_eq!(result, 0);
        assert!(fatal);
    }

    #[test]
    fn test_eval_div_overflow() {
        // i64::MIN / -1 causes overflow
        let e = binary(lit(i64::MIN), Operator::Div, lit(-1));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        let result = naive_eval(&e, &mut x, &mut y, &mut fatal);
        assert_eq!(result, 0);
        assert!(fatal);
    }

    #[test]
    fn test_eval_mod() {
        let e = binary(lit(17), Operator::Mod, lit(5));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 2);
        assert!(!fatal);
    }

    #[test]
    fn test_eval_mod_by_zero() {
        let e = binary(lit(10), Operator::Mod, lit(0));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        let result = naive_eval(&e, &mut x, &mut y, &mut fatal);
        assert_eq!(result, 0);
        assert!(fatal);
    }

    // ==================== Power Operator Tests ====================

    #[test]
    fn test_eval_pow_positive() {
        let e = binary(lit(2), Operator::Pow, lit(10));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 1024);
        assert!(!fatal);
    }

    #[test]
    fn test_eval_pow_zero_exponent() {
        let e = binary(lit(5), Operator::Pow, lit(0));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 1);
        assert!(!fatal);
    }

    #[test]
    fn test_eval_pow_negative_exponent() {
        // 2^-3 should be 0 in integer math
        let e = binary(lit(2), Operator::Pow, lit(-3));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 0);
        assert!(!fatal);
    }

    #[test]
    fn test_eval_pow_one_negative_exponent() {
        // 1^-5 = 1
        let e = binary(lit(1), Operator::Pow, lit(-5));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 1);
        assert!(!fatal);
    }

    #[test]
    fn test_eval_pow_minus_one_even_exponent() {
        // (-1)^-4 = 1
        let e = binary(lit(-1), Operator::Pow, lit(-4));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 1);
        assert!(!fatal);
    }

    #[test]
    fn test_eval_pow_minus_one_odd_exponent() {
        // (-1)^-3 = -1
        let e = binary(lit(-1), Operator::Pow, lit(-3));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), -1);
        assert!(!fatal);
    }

    // ==================== Comparison Operations Tests ====================

    #[test]
    fn test_eval_eq_true() {
        let e = binary(lit(5), Operator::Eq, lit(5));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 1);
    }

    #[test]
    fn test_eval_eq_false() {
        let e = binary(lit(5), Operator::Eq, lit(3));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 0);
    }

    #[test]
    fn test_eval_neq() {
        let e = binary(lit(5), Operator::Neq, lit(3));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 1);
    }

    #[test]
    fn test_eval_lt() {
        let e = binary(lit(3), Operator::Lt, lit(5));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 1);
    }

    #[test]
    fn test_eval_leq() {
        let e1 = binary(lit(3), Operator::Leq, lit(5));
        let e2 = binary(lit(5), Operator::Leq, lit(5));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e1, &mut x, &mut y, &mut fatal), 1);
        assert_eq!(naive_eval(&e2, &mut x, &mut y, &mut fatal), 1);
    }

    #[test]
    fn test_eval_gt() {
        let e = binary(lit(5), Operator::Gt, lit(3));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 1);
    }

    #[test]
    fn test_eval_geq() {
        let e1 = binary(lit(5), Operator::Geq, lit(3));
        let e2 = binary(lit(5), Operator::Geq, lit(5));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e1, &mut x, &mut y, &mut fatal), 1);
        assert_eq!(naive_eval(&e2, &mut x, &mut y, &mut fatal), 1);
    }

    // ==================== Logical Operations Tests ====================

    #[test]
    fn test_eval_or() {
        let e1 = binary(lit(0), Operator::Or, lit(0));
        let e2 = binary(lit(1), Operator::Or, lit(0));
        let e3 = binary(lit(0), Operator::Or, lit(1));
        let e4 = binary(lit(1), Operator::Or, lit(1));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e1, &mut x, &mut y, &mut fatal), 0);
        assert_eq!(naive_eval(&e2, &mut x, &mut y, &mut fatal), 1);
        assert_eq!(naive_eval(&e3, &mut x, &mut y, &mut fatal), 1);
        assert_eq!(naive_eval(&e4, &mut x, &mut y, &mut fatal), 1);
    }

    #[test]
    fn test_eval_and() {
        let e1 = binary(lit(0), Operator::And, lit(0));
        let e2 = binary(lit(1), Operator::And, lit(0));
        let e3 = binary(lit(0), Operator::And, lit(1));
        let e4 = binary(lit(1), Operator::And, lit(1));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e1, &mut x, &mut y, &mut fatal), 0);
        assert_eq!(naive_eval(&e2, &mut x, &mut y, &mut fatal), 0);
        assert_eq!(naive_eval(&e3, &mut x, &mut y, &mut fatal), 0);
        assert_eq!(naive_eval(&e4, &mut x, &mut y, &mut fatal), 1);
    }

    // ==================== Bitwise Operations Tests ====================

    #[test]
    fn test_eval_bitor() {
        let e = binary(lit(0b1010), Operator::BitOr, lit(0b1100));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 0b1110);
    }

    #[test]
    fn test_eval_bitxor() {
        let e = binary(lit(0b1010), Operator::BitXor, lit(0b1100));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 0b0110);
    }

    #[test]
    fn test_eval_bitand() {
        let e = binary(lit(0b1010), Operator::BitAnd, lit(0b1100));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 0b1000);
    }

    #[test]
    fn test_eval_shl() {
        let e = binary(lit(1), Operator::BitShl, lit(4));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 16);
    }

    #[test]
    fn test_eval_shr() {
        let e = binary(lit(16), Operator::BitShr, lit(2));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 4);
    }

    // ==================== Unary Operations Tests ====================

    #[test]
    fn test_eval_neg() {
        let e = unary(Operator::Neg, lit(5));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), -5);
    }

    #[test]
    fn test_eval_bitnot() {
        let e = unary(Operator::BitNot, lit(0));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), -1);
    }

    #[test]
    fn test_eval_not() {
        let e1 = unary(Operator::Not, lit(0));
        let e2 = unary(Operator::Not, lit(5));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e1, &mut x, &mut y, &mut fatal), 1);
        assert_eq!(naive_eval(&e2, &mut x, &mut y, &mut fatal), 0);
    }

    #[test]
    fn test_eval_parens() {
        let e = unary(Operator::Parens, lit(42));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 42);
    }

    // ==================== Variable Tests ====================

    #[test]
    fn test_eval_var() {
        let e = binary(var_x(), Operator::Add, var_y());
        let mut x = 10;
        let mut y = 20;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 30);
    }

    #[test]
    fn test_eval_var_mul() {
        let e = binary(var_x(), Operator::Mul, var_y());
        let mut x = 3;
        let mut y = 7;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 21);
    }

    // ==================== Assignment Tests ====================

    #[test]
    fn test_assignment_eq() {
        let e = binary(var_x(), Operator::AssignEq, lit(5));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 5);
        assert_eq!(x, 5);
    }

    #[test]
    fn test_assignment_add_eq() {
        let e = binary(var_x(), Operator::AddEq, lit(3));
        let mut x = 10;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 13);
        assert_eq!(x, 13);
    }

    #[test]
    fn test_assignment_sub_eq() {
        let e = binary(var_x(), Operator::SubEq, lit(3));
        let mut x = 10;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 7);
        assert_eq!(x, 7);
    }

    #[test]
    fn test_assignment_mul_eq() {
        let e = binary(var_x(), Operator::MulEq, lit(3));
        let mut x = 10;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 30);
        assert_eq!(x, 30);
    }

    #[test]
    fn test_assignment_div_eq() {
        let e = binary(var_x(), Operator::DivEq, lit(2));
        let mut x = 10;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 5);
        assert_eq!(x, 5);
    }

    #[test]
    fn test_assignment_mod_eq() {
        let e = binary(var_x(), Operator::ModEq, lit(3));
        let mut x = 10;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 1);
        assert_eq!(x, 1);
    }

    #[test]
    fn test_assignment_bitor_eq() {
        let e = binary(var_x(), Operator::BitOrEq, lit(0b0011));
        let mut x = 0b1100;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 0b1111);
        assert_eq!(x, 0b1111);
    }

    #[test]
    fn test_assignment_bitand_eq() {
        let e = binary(var_x(), Operator::BitAndEq, lit(0b0011));
        let mut x = 0b1111;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 0b0011);
        assert_eq!(x, 0b0011);
    }

    #[test]
    fn test_assignment_bitxor_eq() {
        let e = binary(var_x(), Operator::BitXorEq, lit(0b0011));
        let mut x = 0b1111;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 0b1100);
        assert_eq!(x, 0b1100);
    }

    #[test]
    fn test_assignment_shl_eq() {
        let e = binary(var_x(), Operator::BitShlEq, lit(2));
        let mut x = 4;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 16);
        assert_eq!(x, 16);
    }

    #[test]
    fn test_assignment_shr_eq() {
        let e = binary(var_x(), Operator::BitShrEq, lit(2));
        let mut x = 16;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 4);
        assert_eq!(x, 4);
    }

    #[test]
    fn test_assignment_to_y() {
        let e = binary(var_y(), Operator::AssignEq, lit(42));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 42);
        assert_eq!(y, 42);
    }

    // ==================== Increment/Decrement Tests ====================

    #[test]
    fn test_pre_inc() {
        let e = unary(Operator::PreInc, var_x());
        let mut x = 5;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 6);
        assert_eq!(x, 6);
    }

    #[test]
    fn test_pre_dec() {
        let e = unary(Operator::PreDec, var_x());
        let mut x = 5;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 4);
        assert_eq!(x, 4);
    }

    #[test]
    fn test_post_inc() {
        let e = unary(Operator::PostInc, var_x());
        let mut x = 5;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 5); // returns old value
        assert_eq!(x, 6);
    }

    #[test]
    fn test_post_dec() {
        let e = unary(Operator::PostDec, var_x());
        let mut x = 5;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 5); // returns old value
        assert_eq!(x, 4);
    }

    #[test]
    fn test_pre_inc_y() {
        let e = unary(Operator::PreInc, var_y());
        let mut x = 0;
        let mut y = 10;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 11);
        assert_eq!(y, 11);
    }

    // ==================== Complex Expression Tests ====================

    #[test]
    fn test_nested_expression() {
        // (x + y) * 2
        let inner = Arc::new(binary(var_x(), Operator::Add, var_y()));
        let e = binary(inner, Operator::Mul, lit(2));
        let mut x = 3;
        let mut y = 5;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 16);
    }

    #[test]
    fn test_deeply_nested() {
        // ((x + 1) * 2) - 3
        let inner1 = Arc::new(binary(var_x(), Operator::Add, lit(1)));
        let inner2 = Arc::new(binary(inner1, Operator::Mul, lit(2)));
        let e = binary(inner2, Operator::Sub, lit(3));
        let mut x = 5;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), 9); // ((5+1)*2)-3 = 9
    }

    // ==================== is_assignment Tests ====================

    #[test]
    fn test_is_assignment() {
        let assign = binary(var_x(), Operator::AssignEq, lit(5));
        let add_eq = binary(var_x(), Operator::AddEq, lit(5));
        let add = binary(var_x(), Operator::Add, lit(5));
        
        assert!(assign.is_assignment());
        assert!(add_eq.is_assignment());
        assert!(!add.is_assignment());
    }

    // ==================== Expression Equality Tests ====================

    #[test]
    fn test_expr_equality_same_literal() {
        let e1 = Expr {
            left: None,
            right: None,
            literal: 5,
            op: Operator::Literal,
            jit: None,
        };
        let e2 = Expr {
            left: None,
            right: None,
            literal: 5,
            op: Operator::Literal,
            jit: None,
        };
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_expr_equality_different_literal() {
        let e1 = Expr {
            left: None,
            right: None,
            literal: 5,
            op: Operator::Literal,
            jit: None,
        };
        let e2 = Expr {
            left: None,
            right: None,
            literal: 10,
            op: Operator::Literal,
            jit: None,
        };
        assert_ne!(e1, e2);
    }

    #[test]
    fn test_expr_equality_semantically_equal() {
        // x + 0 should equal x for semantic equality
        let e1 = binary(var_x(), Operator::Add, lit(0));
        let e2 = Expr {
            left: None,
            right: None,
            literal: 0,
            op: Operator::Var,
            jit: None,
        };
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_expr_equality_mul_by_one() {
        // x * 1 should equal x
        let e1 = binary(var_x(), Operator::Mul, lit(1));
        let e2 = Expr {
            left: None,
            right: None,
            literal: 0,
            op: Operator::Var,
            jit: None,
        };
        assert_eq!(e1, e2);
    }

    // ==================== Wrapping Arithmetic Tests ====================

    #[test]
    fn test_wrapping_add() {
        let e = binary(lit(i64::MAX), Operator::Add, lit(1));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), i64::MIN);
        assert!(!fatal); // wrapping should not be fatal
    }

    #[test]
    fn test_wrapping_sub() {
        let e = binary(lit(i64::MIN), Operator::Sub, lit(1));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), i64::MAX);
        assert!(!fatal);
    }

    #[test]
    fn test_wrapping_neg() {
        let e = unary(Operator::Neg, lit(i64::MIN));
        let mut x = 0;
        let mut y = 0;
        let mut fatal = false;
        assert_eq!(naive_eval(&e, &mut x, &mut y, &mut fatal), i64::MIN);
        assert!(!fatal);
    }
}
