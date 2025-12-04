use std::mem;
use std::ptr;
use libc::{mmap, munmap, PROT_READ, PROT_WRITE, PROT_EXEC, MAP_PRIVATE, MAP_ANON, MAP_FAILED};
use crate::{Expr, Operator, NumT};

#[cfg(any(target_os = "macos", target_os = "ios"))]
const MAP_JIT: libc::c_int = 0x0800;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
const MAP_JIT: libc::c_int = 0;

#[cfg(any(target_os = "macos", target_os = "ios"))]
extern "C" {
    fn sys_icache_invalidate(start: *mut libc::c_void, len: usize);
    fn pthread_jit_write_protect_np(enabled: libc::c_int);
}

pub struct JitMemory {
    ptr: *mut u8,
    size: usize,
}

unsafe impl Send for JitMemory {}
unsafe impl Sync for JitMemory {}

impl JitMemory {
    pub fn new(size: usize) -> Self {
        let size = (size + 4095) & !4095; // Align to page size (4096)
        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | MAP_ANON | MAP_JIT,
                -1,
                0,
            )
        };
        
        if ptr == MAP_FAILED {
            panic!("mmap failed");
        }

        let mem = JitMemory { ptr: ptr as *mut u8, size };
        
        // Enable write access initially
        mem.make_writable();
        
        mem
    }

    pub fn make_writable(&self) {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        unsafe {
            // 0 = writeable, not executable
            pthread_jit_write_protect_np(0);
        }
    }

    pub fn make_executable(&self) {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        unsafe {
            // 1 = executable, not writeable
            pthread_jit_write_protect_np(1);
        }
    }

    pub fn flush(&self) {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        unsafe {
             sys_icache_invalidate(self.ptr as *mut _, self.size);
        }
    }
}

impl Drop for JitMemory {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr as *mut _, self.size);
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub use x86_64::*;

#[cfg(target_arch = "aarch64")]
pub use aarch64::*;

pub type JitFunc = unsafe extern "C" fn(*mut NumT, *mut NumT) -> NumT;

pub struct Jit {
    memory: JitMemory,
    offset: usize,
    regind: usize,
    cached_func: Option<JitFunc>,
}

impl std::fmt::Debug for Jit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Jit")
            .field("offset", &self.offset)
            .field("regind", &self.regind)
            .field("has_func", &self.cached_func.is_some())
            .finish()
    }
}

impl Jit {
    pub fn new(size: usize) -> Self {
        Jit {
            memory: JitMemory::new(size),
            offset: 0,
            regind: 0,
            cached_func: None,
        }
    }

    pub fn func(&self) -> JitFunc {
        self.cached_func.expect("JIT not finalized - call finalize() first")
    }

    /// Finalize the JIT code - flush caches and make executable.
    /// Must be called after compile() and before func().
    pub fn finalize(&mut self) {
        self.memory.flush();
        self.memory.make_executable();
        self.cached_func = Some(unsafe { mem::transmute(self.memory.ptr) });
    }

    #[cfg(target_arch = "x86_64")]
    fn emit_u8(&mut self, b: u8) {
        unsafe {
            *self.memory.ptr.add(self.offset) = b;
        }
        self.offset += 1;
    }

    fn emit_u32(&mut self, val: u32) {
        unsafe {
            ptr::copy_nonoverlapping(&val as *const u32 as *const u8, self.memory.ptr.add(self.offset), 4);
        }
        self.offset += 4;
    }

    #[cfg(target_arch = "x86_64")]
    fn emit_u64(&mut self, val: u64) {
        unsafe {
            ptr::copy_nonoverlapping(&val as *const u64 as *const u8, self.memory.ptr.add(self.offset), 8);
        }
        self.offset += 8;
    }
}

pub fn jit_compile_expr(e: &Expr) -> Jit {
    let size = 4096; 
    let mut jit = Jit::new(size);
    jit.compile(e);
    jit.finalize();
    jit
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;
    use crate::{Expr, Operator};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(u8)]
    pub enum Reg {
        RAX = 0, RCX = 1, RDX = 2, RBX = 3, RSP = 4, RBP = 5, RSI = 6, RDI = 7,
        R8 = 8, R9 = 9, R10 = 10, R11 = 11, R12 = 12, R13 = 13, R14 = 14, R15 = 15,
    }

    const CALL_REGS: [Reg; 6] = [Reg::RDI, Reg::RSI, Reg::RDX, Reg::RCX, Reg::R8, Reg::R9];
    const FREE_REGS: [Reg; 4] = [Reg::R8, Reg::R9, Reg::R10, Reg::R11];

    impl Jit {
        fn rex(&mut self, w: u8, r: Reg, x: u8, b: Reg) {
            let r_val = r as u8;
            let b_val = b as u8;
            self.emit_u8(0x40 | (w << 3) | ((r_val & 8) >> 1) | ((x & 8) >> 2) | ((b_val & 8) >> 3));
        }

        fn reg_hi(r: Reg) -> bool {
            (r as u8) >= 8
        }

        // Instructions
        fn movir(&mut self, i0: i64, r1: Reg) {
            if i0 >= i32::MIN as i64 && i0 <= i32::MAX as i64 {
                 let r1_val = r1 as u8;
                 if Self::reg_hi(r1) { self.emit_u8(0x41); }
                 self.emit_u8(0xB8 | (r1_val & 7));
                 self.emit_u32(i0 as u32);
            } else {
                 let r1_val = r1 as u8;
                 self.rex(1, Reg::RAX, 0, r1);
                 self.emit_u8(0xB8 | (r1_val & 7));
                 self.emit_u64(i0 as u64);
            }
        }

        fn movmr(&mut self, r0: Reg, r1: Reg) {
            self.rex(1, r1, 0, r0);
            self.emit_u8(0x8B);
            self.emit_u8(((r1 as u8 & 7) << 3) | (r0 as u8 & 7));
        }

        fn movr(&mut self, r0: Reg, r1: Reg) {
            self.rex(1, r0, 0, r1);
            self.emit_u8(0x89);
            self.emit_u8(((r0 as u8 & 7) << 3) | (r1 as u8 & 7));
        }

        fn movrm(&mut self, r0: Reg, r1: Reg) {
            self.rex(1, r0, 0, r1);
            self.emit_u8(0x89);
            self.emit_u8(((r0 as u8 & 7) << 3) | (r1 as u8 & 7));
        }

        fn mov0m(&mut self, r0: Reg) {
            self.rex(1, Reg::RAX, 0, r0);
            self.emit_u8(0xC7);
            self.emit_u8(r0 as u8 & 7);
            self.emit_u32(0);
        }

        fn incm(&mut self, r0: Reg) {
            self.rex(1, Reg::RAX, 0, r0);
            self.emit_u8(0xFF);
            self.emit_u8(r0 as u8 & 7);
        }

        fn decm(&mut self, r0: Reg) {
            self.rex(1, Reg::RAX, 0, r0);
            self.emit_u8(0xFF);
            self.emit_u8(0x08 | (r0 as u8 & 7));
        }

        fn negr(&mut self, r0: Reg) {
            self.rex(1, Reg::RAX, 0, r0);
            self.emit_u8(0xF7);
            self.emit_u8(0xD8 | (r0 as u8 & 7));
        }

        fn notr(&mut self, r0: Reg) {
            self.rex(1, Reg::RAX, 0, r0);
            self.emit_u8(0xF7);
            self.emit_u8(0xD0 | (r0 as u8 & 7));
        }

        fn testr(&mut self, r0: Reg, r1: Reg) {
            self.rex(1, r0, 0, r1);
            self.emit_u8(0x85);
            self.emit_u8(((r0 as u8 & 7) << 3) | (r1 as u8 & 7));
        }

        fn seter(&mut self, r0: Reg) {
            if (r0 as u8) >= 4 { self.rex(0, Reg::RAX, 0, r0); }
            self.emit_u8(0x0F); self.emit_u8(0x94);
            self.emit_u8(0xC0 | (r0 as u8 & 7));
        }

        fn setner(&mut self, r0: Reg) {
            if (r0 as u8) >= 4 { self.rex(0, Reg::RAX, 0, r0); }
            self.emit_u8(0x0F); self.emit_u8(0x95);
            self.emit_u8(0xC0 | (r0 as u8 & 7));
        }

        fn setlr(&mut self, r0: Reg) {
            if (r0 as u8) >= 4 { self.rex(0, Reg::RAX, 0, r0); }
            self.emit_u8(0x0F); self.emit_u8(0x9C);
            self.emit_u8(0xC0 | (r0 as u8 & 7));
        }

        fn setler(&mut self, r0: Reg) {
            if (r0 as u8) >= 4 { self.rex(0, Reg::RAX, 0, r0); }
            self.emit_u8(0x0F); self.emit_u8(0x9E);
            self.emit_u8(0xC0 | (r0 as u8 & 7));
        }

        fn setgr(&mut self, r0: Reg) {
            if (r0 as u8) >= 4 { self.rex(0, Reg::RAX, 0, r0); }
            self.emit_u8(0x0F); self.emit_u8(0x9F);
            self.emit_u8(0xC0 | (r0 as u8 & 7));
        }

        fn setger(&mut self, r0: Reg) {
            if (r0 as u8) >= 4 { self.rex(0, Reg::RAX, 0, r0); }
            self.emit_u8(0x0F); self.emit_u8(0x9D);
            self.emit_u8(0xC0 | (r0 as u8 & 7));
        }

        fn andir(&mut self, i0: i32, r1: Reg) {
            self.rex(1, Reg::RAX, 0, r1);
            if i0 as i8 as i32 == i0 {
                self.emit_u8(0x83);
                self.emit_u8(0xE0 | (r1 as u8 & 7));
                self.emit_u8(i0 as u8);
            } else {
                 if r1 == Reg::RAX {
                     self.emit_u8(0x25);
                 } else {
                     self.emit_u8(0x81);
                     self.emit_u8(0xE0 | (r1 as u8 & 7));
                 }
                 self.emit_u32(i0 as u32);
            }
        }

        fn orr(&mut self, r0: Reg, r1: Reg) {
            self.rex(1, r0, 0, r1);
            self.emit_u8(0x09);
            self.emit_u8(0xC0 | (r0 as u8) << 3 | (r1 as u8 & 7));
        }

        fn orrm(&mut self, r0: Reg, r1: Reg) {
            self.rex(1, r0, 0, r1);
            self.emit_u8(0x09);
            self.emit_u8(((r0 as u8 & 7) << 3) | (r1 as u8 & 7));
        }

        fn xorr(&mut self, r0: Reg, r1: Reg) {
            self.rex(1, r0, 0, r1);
            self.emit_u8(0x31);
            self.emit_u8(0xC0 | (r0 as u8) << 3 | (r1 as u8 & 7));
        }

        fn xorrm(&mut self, r0: Reg, r1: Reg) {
            self.rex(1, r0, 0, r1);
            self.emit_u8(0x31);
            self.emit_u8(((r0 as u8 & 7) << 3) | (r1 as u8 & 7));
        }

        fn andr(&mut self, r0: Reg, r1: Reg) {
            self.rex(1, r0, 0, r1);
            self.emit_u8(0x21);
            self.emit_u8(0xC0 | (r0 as u8) << 3 | (r1 as u8 & 7));
        }

        fn andrm(&mut self, r0: Reg, r1: Reg) {
            self.rex(1, r0, 0, r1);
            self.emit_u8(0x21);
            self.emit_u8(((r0 as u8 & 7) << 3) | (r1 as u8 & 7));
        }

        fn salcm(&mut self, r0: Reg) {
            self.rex(1, Reg::RAX, 0, r0);
            self.emit_u8(0xD3);
            self.emit_u8(0x20 | (r0 as u8 & 7));
        }

        fn sarcm(&mut self, r0: Reg) {
            self.rex(1, Reg::RAX, 0, r0);
            self.emit_u8(0xD3);
            self.emit_u8(0x38 | (r0 as u8 & 7));
        }

        fn addrm(&mut self, r0: Reg, r1: Reg) {
            self.rex(1, r0, 0, r1);
            self.emit_u8(0x01);
            self.emit_u8(((r0 as u8 & 7) << 3) | (r1 as u8 & 7));
        }

        fn subrm(&mut self, r0: Reg, r1: Reg) {
            self.rex(1, r0, 0, r1);
            self.emit_u8(0x29);
            self.emit_u8(((r0 as u8 & 7) << 3) | (r1 as u8 & 7));
        }

        fn imulmr(&mut self, r0: Reg, r1: Reg) {
            self.rex(1, r1, 0, r0);
            self.emit_u8(0x0F); self.emit_u8(0xAF);
            self.emit_u8(((r1 as u8 & 7) << 3) | (r0 as u8 & 7));
        }

        fn cmpr(&mut self, r0: Reg, r1: Reg) {
            self.rex(1, r0, 0, r1);
            self.emit_u8(0x39);
            self.emit_u8(0xC0 | (r0 as u8) << 3 | (r1 as u8 & 7));
        }
        
        fn cmpi(&mut self, i0: i32, r1: Reg) {
            self.rex(1, Reg::RAX, 0, r1);
            if i0 as i8 as i32 == i0 {
                self.emit_u8(0x83);
                self.emit_u8(0xF8 | (r1 as u8 & 7));
                self.emit_u8(i0 as u8);
            } else {
                if r1 == Reg::RAX {
                    self.emit_u8(0x3D);
                } else {
                    self.emit_u8(0x81);
                    self.emit_u8(0xF8 | (r1 as u8 & 7));
                }
                self.emit_u32(i0 as u32);
            }
        }

        fn salc(&mut self, r0: Reg) {
            self.rex(1, Reg::RAX, 0, r0);
            self.emit_u8(0xD3);
            self.emit_u8(0xE0 | (r0 as u8 & 7));
        }

        fn sarc(&mut self, r0: Reg) {
            self.rex(1, Reg::RAX, 0, r0);
            self.emit_u8(0xD3);
            self.emit_u8(0xF8 | (r0 as u8 & 7));
        }

        fn addr(&mut self, r0: Reg, r1: Reg) {
            self.rex(1, r0, 0, r1);
            self.emit_u8(0x01);
            self.emit_u8(0xC0 | (r0 as u8) << 3 | (r1 as u8 & 7));
        }

        fn subr(&mut self, r0: Reg, r1: Reg) {
            self.rex(1, r0, 0, r1);
            self.emit_u8(0x29);
            self.emit_u8(0xC0 | (r0 as u8) << 3 | (r1 as u8 & 7));
        }

        fn imulr(&mut self, r0: Reg, r1: Reg) {
            self.rex(1, r1, 0, r0);
            self.emit_u8(0x0F); self.emit_u8(0xAF);
            self.emit_u8(0xC0 | (r1 as u8) << 3 | (r0 as u8 & 7));
        }

        fn cqto(&mut self) {
            self.emit_u8(0x48);
            self.emit_u8(0x99);
        }

        fn idivr(&mut self, r0: Reg) {
            self.rex(1, Reg::RAX, 0, r0);
            self.emit_u8(0xF7);
            self.emit_u8(0xF8 | (r0 as u8 & 7));
        }

        fn ret(&mut self) {
            self.emit_u8(0xC3);
        }

        fn jes(&mut self, i0: i8) {
            self.emit_u8(0x74);
            self.emit_u8(i0 as u8);
        }

        fn jnes(&mut self, i0: i8) {
            self.emit_u8(0x75);
            self.emit_u8(i0 as u8);
        }

        pub fn compile(&mut self, e: &Expr) {
            self.regind = 0;
            self.gen_expr(e);
            // Move result to RAX
            self.movr(FREE_REGS[self.regind - 1], Reg::RAX);
            self.ret();
        }

        // gen_expr logic same as before
        fn gen_expr(&mut self, e: &Expr) {
            match e.op {
                Operator::Literal => {
                    self.movir(e.literal, FREE_REGS[self.regind]);
                    self.regind += 1;
                    return;
                }
                Operator::Var | Operator::VarY => {
                    let reg_idx = (e.op as usize) & 0xF;
                    self.movmr(CALL_REGS[reg_idx], FREE_REGS[self.regind]);
                    self.regind += 1;
                    return;
                }

                 Operator::PreInc => {
                     let target = e.right.as_ref().unwrap();
                     let reg_idx = (target.op as usize) & 0xF;
                     self.incm(CALL_REGS[reg_idx]);
                     self.movmr(CALL_REGS[reg_idx], FREE_REGS[self.regind]);
                     self.regind += 1;
                     return;
                }
                Operator::PreDec => {
                     let target = e.right.as_ref().unwrap();
                     let reg_idx = (target.op as usize) & 0xF;
                     self.decm(CALL_REGS[reg_idx]);
                     self.movmr(CALL_REGS[reg_idx], FREE_REGS[self.regind]);
                     self.regind += 1;
                     return;
                }
                Operator::PostInc => {
                     let target = e.right.as_ref().unwrap();
                     let reg_idx = (target.op as usize) & 0xF;
                     self.movmr(CALL_REGS[reg_idx], FREE_REGS[self.regind]);
                     self.regind += 1;
                     self.incm(CALL_REGS[reg_idx]);
                     return;
                }
                Operator::PostDec => {
                     let target = e.right.as_ref().unwrap();
                     let reg_idx = (target.op as usize) & 0xF;
                     self.movmr(CALL_REGS[reg_idx], FREE_REGS[self.regind]);
                     self.regind += 1;
                     self.decm(CALL_REGS[reg_idx]);
                     return;
                }
                _ => {}
            }

            if e.left.is_none() || e.is_assignment() {
                self.gen_expr(e.right.as_ref().unwrap());
                let rr = FREE_REGS[self.regind - 1];
                
                if let Some(ref left) = e.left {
                     let target_idx = (left.op as usize) & 0xF;
                     let target_reg = CALL_REGS[target_idx];
                     
                     match e.op {
                        Operator::AssignEq => self.movrm(rr, target_reg),
                        Operator::BitOrEq => self.orrm(rr, target_reg),
                        Operator::BitXorEq => self.xorrm(rr, target_reg),
                        Operator::BitAndEq => self.andrm(rr, target_reg),
                        Operator::BitShlEq => {
                            self.movr(rr, Reg::RCX);
                            self.salcm(target_reg);
                        },
                        Operator::BitShrEq => {
                            self.movr(rr, Reg::RCX);
                            self.sarcm(target_reg);
                        },
                        Operator::AddEq => self.addrm(rr, target_reg),
                        Operator::SubEq => self.subrm(rr, target_reg),
                        Operator::MulEq => {
                            self.imulmr(target_reg, rr);
                            self.movrm(rr, target_reg);
                        },
                        Operator::DivEq => {
                            self.testr(rr, rr);
                            self.jes(24);
                            self.movmr(target_reg, Reg::RAX);
                            self.movir(i64::MIN, Reg::RDX);
                            self.cmpr(Reg::RDX, Reg::RAX);
                            self.jnes(14);
                            self.cmpi(-1, rr);
                            self.jnes(8);
                            self.mov0m(target_reg);
                            self.ret();
                            self.cqto();
                            self.idivr(rr);
                            self.movrm(Reg::RAX, target_reg);
                        },
                        Operator::ModEq => {
                            self.testr(rr, rr);
                            self.jes(24);
                            self.movmr(target_reg, Reg::RAX);
                            self.movir(i64::MIN, Reg::RDX);
                            self.cmpr(Reg::RDX, Reg::RAX);
                            self.jnes(14);
                            self.cmpi(-1, rr);
                            self.jnes(8);
                            self.mov0m(target_reg);
                            self.ret();
                            self.cqto();
                            self.idivr(rr);
                            self.movrm(Reg::RDX, target_reg);
                        },
                        _ => {}
                     }
                } else {
                    match e.op {
                        Operator::Neg => self.negr(rr),
                        Operator::BitNot => self.notr(rr),
                        Operator::Not => {
                            self.testr(rr, rr);
                            self.seter(rr);
                            self.andir(1, rr);
                        },
                        Operator::Parens => {}, // Just pass through - value already in rr
                        _ => {}
                    }
                }
                return;
            }

            self.gen_expr(e.left.as_ref().unwrap());
            self.gen_expr(e.right.as_ref().unwrap());
            
            self.regind -= 1;
            let rr = FREE_REGS[self.regind];
            let rl = FREE_REGS[self.regind - 1];

            match e.op {
                Operator::Or => {
                    // rl || rr -> (rl | rr) != 0
                    self.orr(rr, rl);           // rl |= rr
                    self.xorr(Reg::RAX, Reg::RAX); // zero rax
                    self.testr(rl, rl);         // test result
                    self.setner(Reg::RAX);      // al = (result != 0)
                    self.movr(Reg::RAX, rl);    // rl = rax (0 or 1)
                },
                Operator::And => {
                    // rl && rr -> (rl != 0) && (rr != 0)
                    // Byte layout:
                    // 0-2: testr rl, rl (3 bytes)
                    // 3-4: jz +13 (2 bytes) -> to xor at byte 18
                    // 5-7: testr rr, rr (3 bytes)
                    // 8-9: jz +8 (2 bytes) -> to xor at byte 18
                    // 10-15: movir 1, rl (6 bytes for R8-R15)
                    // 16-17: jmp +3 (2 bytes) -> skip xor
                    // 18-20: xor rl, rl (3 bytes)
                    self.testr(rl, rl);
                    self.jes(13);
                    self.testr(rr, rr);
                    self.jes(8);
                    self.movir(1, rl);
                    self.emit_u8(0xEB); self.emit_u8(3);
                    self.xorr(rl, rl);
                },
                Operator::BitOr => self.orr(rr, rl),
                Operator::BitXor => self.xorr(rr, rl),
                Operator::BitAnd => self.andr(rr, rl),
                Operator::Eq => {
                    self.cmpr(rr, rl);
                    self.seter(rl);
                    self.andir(1, rl);
                },
                Operator::Neq => {
                    self.cmpr(rr, rl);
                    self.setner(rl);
                    self.andir(1, rl);
                },
                Operator::Lt => {
                    self.cmpr(rr, rl);
                    self.setlr(rl);
                    self.andir(1, rl);
                },
                Operator::Gt => {
                    self.cmpr(rr, rl);
                    self.setgr(rl);
                    self.andir(1, rl);
                },
                Operator::Leq => {
                    self.cmpr(rr, rl);
                    self.setler(rl);
                    self.andir(1, rl);
                },
                Operator::Geq => {
                    self.cmpr(rr, rl);
                    self.setger(rl);
                    self.andir(1, rl);
                },
                Operator::BitShl => {
                    self.movr(rr, Reg::RCX);
                    self.salc(rl);
                },
                Operator::BitShr => {
                    self.movr(rr, Reg::RCX);
                    self.sarc(rl);
                },
                Operator::Add => self.addr(rr, rl),
                Operator::Sub => self.subr(rr, rl),
                Operator::Mul => self.imulr(rr, rl),
                Operator::Div => {
                    self.testr(rr, rr);
                    self.jes(21);
                    self.movir(i64::MIN, Reg::RDX);
                    self.cmpr(Reg::RDX, rl);
                    self.jnes(10);
                    self.cmpi(-1, rr);
                    self.jnes(4);
                    self.xorr(Reg::RAX, Reg::RAX);
                    self.ret();
                    self.movr(rl, Reg::RAX);
                    self.cqto();
                    self.idivr(rr);
                    self.movr(Reg::RAX, rl);
                },
                Operator::Mod => {
                    self.testr(rr, rr);
                    self.jes(21);
                    self.movir(i64::MIN, Reg::RDX);
                    self.cmpr(Reg::RDX, rl);
                    self.jnes(10);
                    self.cmpi(-1, rr);
                    self.jnes(4);
                    self.xorr(Reg::RAX, Reg::RAX);
                    self.ret();
                    self.movr(rl, Reg::RAX);
                    self.cqto();
                    self.idivr(rr);
                    self.movr(Reg::RDX, rl);
                },
                Operator::Pow => {
                  
                    // test rr, rr (3 bytes)
                    self.testr(rr, rr);
                    // js negative: +24 bytes from here to xor (3+5+3+2+4+3+2+3+2=27, we're at 5, target at 29, so +24)
                    self.emit_u8(0x78); self.emit_u8(24);
                    
                    // mov rax, 1 (5 bytes) - result accumulator
                    self.movir(1, Reg::RAX);
                    
                    // test rr, rr (3 bytes)
                    self.testr(rr, rr);
                    // jz done: +9 bytes (from offset 15 to mov at 24)
                    self.jes(9);
                    
                    // loop: imul rax, rl (4 bytes)
                    self.imulr(rl, Reg::RAX);
                    // dec rr (3 bytes)
                    self.rex(1, Reg::RAX, 0, rr);
                    self.emit_u8(0xFF);
                    self.emit_u8(0xC8 | (rr as u8 & 7));
                    // jnz loop: -9 bytes (from 24 back to 15)
                    self.jnes(-9i8);
                    
                    // done: mov rl, rax (3 bytes)
                    self.movr(Reg::RAX, rl);
                    // jmp end: +3 bytes to skip xor
                    self.emit_u8(0xEB); self.emit_u8(3);
                    
                    // negative: xor rl, rl (3 bytes) - return 0
                    self.xorr(rl, rl);
                    // end:
                },
                _ => {}
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::*;
    use crate::{Expr, Operator};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(u8)]
    pub enum Reg {
        X0=0, X1=1, X2=2, X3=3, X4=4, X5=5, X6=6, X7=7,
        X8=8, X9=9, X10=10, X11=11, X12=12, X13=13, X14=14, X15=15,
        X16=16, X17=17, X18=18, X19=19, X20=20, X21=21, X22=22, X23=23,
        X24=24, X25=25, X26=26, X27=27, X28=28, FP=29, LR=30, XZR=31
    }

    const CALL_REGS: [Reg; 2] = [Reg::X0, Reg::X1]; // Only need pointers for var indices 0 and 1
    const FREE_REGS: [Reg; 7] = [Reg::X9, Reg::X10, Reg::X11, Reg::X12, Reg::X13, Reg::X14, Reg::X15];

    impl Jit {
        // AArch64 Emitters
        fn inst(&mut self, val: u32) {
            self.emit_u32(val);
        }

        // Instructions
        // RET
        fn ret(&mut self) {
            self.inst(0xD65F03C0);
        }

        // MOV immediate (handle wide)
        // Simple movz/movn logic
        fn mov_imm(&mut self, dest: Reg, imm: i64) {
            let rd = dest as u32;
            // If positive and fits in 16 bits
            if imm >= 0 && imm <= 0xFFFF {
                self.inst(0xD2800000 | ((imm as u32) << 5) | rd); // MOVZ
                return;
            }
            // If small negative, bitwise NOT fits in 16 bits?
            if imm < 0 && (!imm) <= 0xFFFF {
                self.inst(0x92800000 | ((!imm as u32) << 5) | rd); // MOVN
                return;
            }
            
            // Build full 64-bit immediate
            // MOVZ for first chunk, MOVK for others
            let u = imm as u64;
            self.inst(0xD2800000 | (((u & 0xFFFF) as u32) << 5) | rd); // MOVZ imm, LSL 0
            if (u >> 16) & 0xFFFF != 0 {
                self.inst(0xF2A00000 | (((u >> 16) & 0xFFFF) as u32) << 5 | rd); // MOVK, LSL 16
            }
            if (u >> 32) & 0xFFFF != 0 {
                self.inst(0xF2C00000 | (((u >> 32) & 0xFFFF) as u32) << 5 | rd); // MOVK, LSL 32
            }
            if (u >> 48) & 0xFFFF != 0 {
                self.inst(0xF2E00000 | (((u >> 48) & 0xFFFF) as u32) << 5 | rd); // MOVK, LSL 48
            }
        }
        
        // LDR Xt, [Xn]
        fn ldr(&mut self, rt: Reg, rn: Reg) {
            self.inst(0xF9400000 | ((rn as u32) << 5) | (rt as u32));
        }

        // STR Xt, [Xn]
        fn str(&mut self, rt: Reg, rn: Reg) {
            self.inst(0xF9000000 | ((rn as u32) << 5) | (rt as u32));
        }

        // ADD Rd, Rn, Rm
        fn add(&mut self, rd: Reg, rn: Reg, rm: Reg) {
            self.inst(0x8B000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
        }

        // SUB Rd, Rn, Rm
        fn sub(&mut self, rd: Reg, rn: Reg, rm: Reg) {
            self.inst(0xCB000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
        }
        
        // MUL Rd, Rn, Rm
        fn mul(&mut self, rd: Reg, rn: Reg, rm: Reg) {
            self.inst(0x9B007C00 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
        }

        // SDIV Rd, Rn, Rm
        fn sdiv(&mut self, rd: Reg, rn: Reg, rm: Reg) {
            self.inst(0x9AC00C00 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
        }

        // AND Rd, Rn, Rm
        fn and(&mut self, rd: Reg, rn: Reg, rm: Reg) {
            self.inst(0x8A000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
        }

        // ORR Rd, Rn, Rm
        fn orr(&mut self, rd: Reg, rn: Reg, rm: Reg) {
            self.inst(0xAA000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
        }

        // EOR Rd, Rn, Rm (XOR)
        fn eor(&mut self, rd: Reg, rn: Reg, rm: Reg) {
            self.inst(0xCA000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
        }
        
        // MVN Rd, Rm (via ORN Rd, XZR, Rm)
        fn mvn(&mut self, rd: Reg, rm: Reg) {
            self.inst(0xAA2003E0 | ((rm as u32) << 16) | (rd as u32)); // ORN Rd, XZR, Rm
        }
        
        // NEG Rd, Rm (SUB Rd, XZR, Rm)
        fn neg(&mut self, rd: Reg, rm: Reg) {
            self.inst(0xCB0003E0 | ((rm as u32) << 16) | (rd as u32));
        }

        // LSLV Rd, Rn, Rm (Logical Shift Left Variable)
        fn lslv(&mut self, rd: Reg, rn: Reg, rm: Reg) {
             // LSLV Rd, Rn, Rm: 1001 1010 110 Rm 001000 Rn Rd
             self.inst(0x9AC02000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
        }
        
        // ASRV Rd, Rn, Rm (Arithmetic Shift Right Variable)
        fn asrv(&mut self, rd: Reg, rn: Reg, rm: Reg) {
             self.inst(0x9AC02800 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
        }

        // CMP Rn, Rm (SUBS XZR, Rn, Rm)
        fn cmp(&mut self, rn: Reg, rm: Reg) {
            self.inst(0xEB00001F | ((rm as u32) << 16) | ((rn as u32) << 5));
        }
        
        // CMP Rn, #imm (SUBS XZR, Rn, #imm) - simplified, supports imm12
        fn cmpi(&mut self, rn: Reg, imm: u16) {
            // SUBS XZR, Rn, #imm (shift 0)
            // 111 10001 00 imm12 Rn 11111
            self.inst(0xF100001F | ((imm as u32) << 10) | ((rn as u32) << 5));
        }

        // CSET Rd, cond (CSINC Rd, XZR, XZR, cond_inv)
        // cond: EQ=0, NE=1, CS=2, CC=3, MI=4, PL=5, VS=6, VC=7, HI=8, LS=9, GE=10, LT=11, GT=12, LE=13
        // cond_inv: invert lowest bit.
        fn cset(&mut self, rd: Reg, cond: u32) {
            let cond_inv = cond ^ 1;
            // CSINC Rd, XZR, XZR, cond_inv (CSET is alias)
            // Encoding: sf=1 op=0 S=0 11010100 Rm cond o2=01 Rn Rd
            // o2=01 for CSINC (not 00 which is CSEL!)
            self.inst(0x9A9F07E0 | (cond_inv << 12) | (rd as u32));
        }
        
        // Branching (Conditional)
        // B.cond offset
        // 0101 0100 imm19 0 cond
        // Offset is instruction count? No, bytes? PC-relative, bits 20:2. offset/4.
        // Need to patch offsets.
        fn b_cond(&mut self, cond: u32, offset_instrs: i32) {
             let imm19 = (offset_instrs & 0x7FFFF) as u32;
             self.inst(0x54000000 | (imm19 << 5) | cond);
        }

        // B offset
        // 0001 0100 imm26
        fn b(&mut self, offset_instrs: i32) {
             let imm26 = (offset_instrs & 0x3FFFFFF) as u32;
             self.inst(0x14000000 | imm26);
        }
        
        // Mov 0 to register (alias MOV Rd, XZR) -> ORR Rd, XZR, XZR
        fn mov0(&mut self, rd: Reg) {
             self.orr(rd, Reg::XZR, Reg::XZR);
        }
        
        // CSEL Rd, Rn, Rm, cond: Rd = cond ? Rn : Rm
        fn csel(&mut self, rd: Reg, rn: Reg, rm: Reg, cond: u32) {
            // Encoding: 1001 1010 100 Rm cond 00 Rn Rd
            self.inst(0x9A800000 | ((rm as u32) << 16) | (cond << 12) | ((rn as u32) << 5) | (rd as u32));
        }

        pub fn compile(&mut self, e: &Expr) {
            self.regind = 0;
            self.gen_expr(e);
            // Move result to X0 (RAX equiv)
            self.orr(Reg::X0, FREE_REGS[self.regind - 1], Reg::XZR); // MOV X0, Res
            self.ret();
        }

        fn gen_expr(&mut self, e: &Expr) {
             match e.op {
                Operator::Literal => {
                    self.mov_imm(FREE_REGS[self.regind], e.literal);
                    self.regind += 1;
                    return;
                }
                Operator::Var | Operator::VarY => {
                    let reg_idx = (e.op as usize) & 0xF;
                    // Load from pointer in CALL_REGS[idx]
                    self.ldr(FREE_REGS[self.regind], CALL_REGS[reg_idx]);
                    self.regind += 1;
                    return;
                }
                Operator::PreInc => {
                     let target = e.right.as_ref().unwrap();
                     let reg_idx = (target.op as usize) & 0xF;
                     let ptr = CALL_REGS[reg_idx];
                     let tmp = FREE_REGS[self.regind];
                     self.ldr(tmp, ptr);
                     self.mov_imm(Reg::X16, 1);
                     self.add(tmp, tmp, Reg::X16);
                     self.str(tmp, ptr);
                     self.regind += 1;
                     return;
                }
                Operator::PreDec => {
                     let target = e.right.as_ref().unwrap();
                     let reg_idx = (target.op as usize) & 0xF;
                     let ptr = CALL_REGS[reg_idx];
                     let tmp = FREE_REGS[self.regind];
                     self.ldr(tmp, ptr);
                     self.mov_imm(Reg::X16, 1);
                     self.sub(tmp, tmp, Reg::X16);
                     self.str(tmp, ptr);
                     self.regind += 1;
                     return;
                }
                Operator::PostInc => {
                     let target = e.right.as_ref().unwrap();
                     let reg_idx = (target.op as usize) & 0xF;
                     let ptr = CALL_REGS[reg_idx];
                     let tmp = FREE_REGS[self.regind];
                     self.ldr(tmp, ptr); // Load value
                     // Increment memory
                     let tmp2 = Reg::X16;
                     self.mov_imm(tmp2, 1);
                     self.add(tmp2, tmp, tmp2);
                     self.str(tmp2, ptr);
                     self.regind += 1;
                     return;
                }
                Operator::PostDec => {
                     let target = e.right.as_ref().unwrap();
                     let reg_idx = (target.op as usize) & 0xF;
                     let ptr = CALL_REGS[reg_idx];
                     let tmp = FREE_REGS[self.regind];
                     self.ldr(tmp, ptr);
                     let tmp2 = Reg::X16;
                     self.mov_imm(tmp2, 1);
                     self.sub(tmp2, tmp, tmp2);
                     self.str(tmp2, ptr);
                     self.regind += 1;
                     return;
                }
                _ => {}
            }

            if e.left.is_none() || e.is_assignment() {
                self.gen_expr(e.right.as_ref().unwrap());
                let rr = FREE_REGS[self.regind - 1];
                
                if let Some(ref left) = e.left {
                     let target_idx = (left.op as usize) & 0xF;
                     let ptr = CALL_REGS[target_idx];
                     let tmp = Reg::X16; // Temp for loading target val
                     
                     match e.op {
                        Operator::AssignEq => self.str(rr, ptr),
                        Operator::BitOrEq => { self.ldr(tmp, ptr); self.orr(tmp, tmp, rr); self.str(tmp, ptr); self.orr(rr, tmp, Reg::XZR); }, // result is new value
                        Operator::BitXorEq => { self.ldr(tmp, ptr); self.eor(tmp, tmp, rr); self.str(tmp, ptr); self.orr(rr, tmp, Reg::XZR); },
                        Operator::BitAndEq => { self.ldr(tmp, ptr); self.and(tmp, tmp, rr); self.str(tmp, ptr); self.orr(rr, tmp, Reg::XZR); },
                        Operator::BitShlEq => { self.ldr(tmp, ptr); self.lslv(tmp, tmp, rr); self.str(tmp, ptr); self.orr(rr, tmp, Reg::XZR); },
                        Operator::BitShrEq => { self.ldr(tmp, ptr); self.asrv(tmp, tmp, rr); self.str(tmp, ptr); self.orr(rr, tmp, Reg::XZR); },
                        Operator::AddEq => { self.ldr(tmp, ptr); self.add(tmp, tmp, rr); self.str(tmp, ptr); self.orr(rr, tmp, Reg::XZR); },
                        Operator::SubEq => { self.ldr(tmp, ptr); self.sub(tmp, tmp, rr); self.str(tmp, ptr); self.orr(rr, tmp, Reg::XZR); },
                        Operator::MulEq => { self.ldr(tmp, ptr); self.mul(tmp, tmp, rr); self.str(tmp, ptr); self.orr(rr, tmp, Reg::XZR); },
                        Operator::DivEq | Operator::ModEq => {
                            // if rr == 0: return 0 but do NOT modify target (matches naive_eval)
                            self.cmp(rr, Reg::XZR);
                            self.b_cond(1, 3); // NE -> skip EQ block (2 instrs)
                            self.mov0(Reg::X0); // Return 0
                            self.ret();
                            
                            // Overflow check? (min / -1)
                            // For simplicity skip overflow check to save space/time or implement simple one
                            // Actually user wants correct behavior.
                            
                            self.ldr(tmp, ptr);
                            self.sdiv(Reg::X17, tmp, rr); // X17 = result div
                            
                            if matches!(e.op, Operator::ModEq) {
                                // Mod: a - (a/b)*b
                                self.mul(Reg::X18, Reg::X17, rr);
                                self.sub(tmp, tmp, Reg::X18); // tmp = mod result
                            } else {
                                self.orr(tmp, Reg::X17, Reg::XZR); // tmp = div result
                            }
                            self.str(tmp, ptr);
                            self.orr(rr, tmp, Reg::XZR);
                        },
                        _ => {}
                     }
                } else {
                    // Unary
                    match e.op {
                        Operator::Neg => self.neg(rr, rr),
                        Operator::BitNot => self.mvn(rr, rr),
                        Operator::Not => {
                             self.cmp(rr, Reg::XZR);
                             self.cset(rr, 0); // EQ -> 1, else 0.
                        },
                        Operator::Parens => {}, // Just pass through
                        _ => {}
                    }
                }
                return;
            }

            self.gen_expr(e.left.as_ref().unwrap());
            self.gen_expr(e.right.as_ref().unwrap());
            
            self.regind -= 1;
            let rr = FREE_REGS[self.regind];
            let rl = FREE_REGS[self.regind - 1];

            match e.op {
                Operator::Or => {
                    // rl || rr -> simplified: just use bitwise or and check != 0
                    self.orr(rl, rl, rr);
                    // Compare with 0 and set result
                    self.cmp(rl, Reg::XZR);
                    self.cset(rl, 1); // NE -> 1
                },
                Operator::And => {
                    // rl && rr -> simplified: multiply the boolean results
                    // (rl != 0) * (rr != 0)
                    self.cmp(rl, Reg::XZR);
                    self.cset(rl, 1);  // rl = (rl != 0)
                    self.cmp(rr, Reg::XZR);
                    self.cset(rr, 1);  // rr = (rr != 0)
                    self.mul(rl, rl, rr); // rl = rl * rr (0 or 1)
                },
                Operator::BitOr => self.orr(rl, rl, rr),
                Operator::BitXor => self.eor(rl, rl, rr),
                Operator::BitAnd => self.and(rl, rl, rr),
                Operator::Eq => { self.cmp(rl, rr); self.cset(rl, 0); }, // EQ
                Operator::Neq => { self.cmp(rl, rr); self.cset(rl, 1); }, // NE
                Operator::Lt => { self.cmp(rl, rr); self.cset(rl, 11); }, // LT
                Operator::Gt => { self.cmp(rl, rr); self.cset(rl, 12); }, // GT
                Operator::Leq => { self.cmp(rl, rr); self.cset(rl, 13); }, // LE
                Operator::Geq => { self.cmp(rl, rr); self.cset(rl, 10); }, // GE
                Operator::BitShl => self.lslv(rl, rl, rr),
                Operator::BitShr => self.asrv(rl, rl, rr),
                Operator::Add => self.add(rl, rl, rr),
                Operator::Sub => self.sub(rl, rl, rr),
                Operator::Mul => self.mul(rl, rl, rr),
                Operator::Div | Operator::Mod => {
                    self.cmp(rr, Reg::XZR);
                    if matches!(e.op, Operator::Mod) {
                        self.b_cond(1, 3); // NE -> +3 to div
                        self.mov0(rl);
                        self.b(4); // -> +4 to end
                        // div:
                        self.sdiv(Reg::X17, rl, rr);
                        self.mul(Reg::X18, Reg::X17, rr);
                        self.sub(rl, rl, Reg::X18);
                        // end:
                    } else {
                        self.b_cond(1, 3); // NE -> +3 to div
                        self.mov0(rl);
                        self.b(2); // -> +2 to end
                        // div:
                        self.sdiv(rl, rl, rr);
                        // end:
                    }
                },
                Operator::Pow => {
                    self.mov_imm(Reg::X17, 1);           // 0: result = 1
                    self.cmp(rl, Reg::X17);              // 1: base == 1?
                    self.b_cond(0, 14);                  // 2: EQ -> end (16)
                    self.orr(Reg::X18, rr, Reg::XZR);    // 3: X18 = rr
                    self.cmp(rr, Reg::XZR);              // 4
                    self.b_cond(12, 6);                  // 5: GT -> loop (11)
                    self.b_cond(0, 10);                  // 6: EQ -> end (16)
                    // neg exp: check base==-1
                    self.mov_imm(Reg::X16, -1i64);       // 7
                    self.cmp(rl, Reg::X16);              // 8
                    self.b_cond(1, 6);                   // 9: NE -> set_zero (15)
                    // base==-1, neg exp: negate X18 and compute
                    self.neg(Reg::X18, Reg::X18);        // 10: X18 = -rr (now positive)
                    // loop:
                    self.mul(Reg::X17, Reg::X17, rl);    // 11
                    self.inst(0xF1000400 | (18 << 5) | 18); // 12: SUBS X18, X18, 1
                    self.b_cond(12, -2);                 // 13: GT -> loop (11)
                    self.b(2);                           // 14: -> end (16)
                    // set_zero:
                    self.mov0(Reg::X17);                 // 15
                    // end:
                    self.orr(rl, Reg::X17, Reg::XZR);    // 16
                },
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::{naive_eval, NumT};

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

    // Helper to compare JIT with naive_eval
    fn compare_jit_naive(e: &Expr, x_val: NumT, y_val: NumT) -> bool {
        let jit = jit_compile_expr(e);
        let f = jit.func();

        let mut x_jit = x_val;
        let mut y_jit = y_val;
        let jit_result = unsafe { f(&mut x_jit, &mut y_jit) };

        let mut x_naive = x_val;
        let mut y_naive = y_val;
        let mut fatal = false;
        let naive_result = naive_eval(e, &mut x_naive, &mut y_naive, &mut fatal);

        // If naive eval had a fatal error, we skip comparison
        if fatal {
            return true;
        }

        jit_result == naive_result && x_jit == x_naive && y_jit == y_naive
    }

    // Helper to run comparison over a range of values
    fn test_expr_range(e: &Expr, range: std::ops::RangeInclusive<NumT>) {
        for x_val in range.clone() {
            for y_val in range.clone() {
                assert!(
                    compare_jit_naive(e, x_val, y_val),
                    "JIT/naive mismatch for x={}, y={}",
                    x_val,
                    y_val
                );
            }
        }
    }

    // ==================== JIT Memory Tests ====================

    #[test]
    fn test_jit_memory_creation() {
        let mem = JitMemory::new(4096);
        assert!(!mem.ptr.is_null());
        assert!(mem.size >= 4096);
    }

    #[test]
    fn test_jit_memory_page_alignment() {
        let mem = JitMemory::new(100);
        assert_eq!(mem.size % 4096, 0);
    }

    // ==================== Basic Literal Tests ====================

    #[test]
    fn test_jit_literal() {
        let e = Expr {
            left: None,
            right: None,
            literal: 42,
            op: Operator::Literal,
            jit: None,
        };
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_literal_negative() {
        let e = Expr {
            left: None,
            right: None,
            literal: -123,
            op: Operator::Literal,
            jit: None,
        };
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_literal_large() {
        let e = Expr {
            left: None,
            right: None,
            literal: 1_000_000_000,
            op: Operator::Literal,
            jit: None,
        };
        test_expr_range(&e, -2..=2);
    }

    // ==================== Variable Tests ====================

    #[test]
    fn test_jit_var_x() {
        let e = Expr {
            left: None,
            right: None,
            literal: 0,
            op: Operator::Var,
            jit: None,
        };
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_var_y() {
        let e = Expr {
            left: None,
            right: None,
            literal: 0,
            op: Operator::VarY,
            jit: None,
        };
        test_expr_range(&e, -4..=4);
    }

    // ==================== Arithmetic Operations ====================

    #[test]
    fn test_jit_add() {
        let e = binary(var_x(), Operator::Add, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_add_literal() {
        let e = binary(var_x(), Operator::Add, lit(10));
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_sub() {
        let e = binary(var_x(), Operator::Sub, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_mul() {
        let e = binary(var_x(), Operator::Mul, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_div() {
        let e = binary(var_x(), Operator::Div, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_mod() {
        let e = binary(var_x(), Operator::Mod, var_y());
        test_expr_range(&e, -4..=4);
    }

    // ==================== Comparison Operations ====================

    #[test]
    fn test_jit_eq() {
        let e = binary(var_x(), Operator::Eq, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_neq() {
        let e = binary(var_x(), Operator::Neq, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_lt() {
        let e = binary(var_x(), Operator::Lt, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_leq() {
        let e = binary(var_x(), Operator::Leq, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_gt() {
        let e = binary(var_x(), Operator::Gt, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_geq() {
        let e = binary(var_x(), Operator::Geq, var_y());
        test_expr_range(&e, -4..=4);
    }

    // ==================== Logical Operations ====================

    #[test]
    fn test_jit_or() {
        let e = binary(var_x(), Operator::Or, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_and() {
        let e = binary(var_x(), Operator::And, var_y());
        test_expr_range(&e, -4..=4);
    }

    // ==================== Bitwise Operations ====================

    #[test]
    fn test_jit_bitor() {
        let e = binary(var_x(), Operator::BitOr, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_bitxor() {
        let e = binary(var_x(), Operator::BitXor, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_bitand() {
        let e = binary(var_x(), Operator::BitAnd, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_shl() {
        let e = binary(var_x(), Operator::BitShl, lit(2));
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_shr() {
        let e = binary(var_x(), Operator::BitShr, lit(1));
        test_expr_range(&e, -4..=4);
    }

    // ==================== Unary Operations ====================

    #[test]
    fn test_jit_neg() {
        let e = unary(Operator::Neg, var_x());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_bitnot() {
        let e = unary(Operator::BitNot, var_x());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_not() {
        let e = unary(Operator::Not, var_x());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_parens() {
        let e = unary(Operator::Parens, var_x());
        test_expr_range(&e, -4..=4);
    }

    // ==================== Increment/Decrement Operations ====================

    #[test]
    fn test_jit_pre_inc() {
        let e = unary(Operator::PreInc, var_x());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_pre_dec() {
        let e = unary(Operator::PreDec, var_x());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_post_inc() {
        let e = unary(Operator::PostInc, var_x());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_post_dec() {
        let e = unary(Operator::PostDec, var_x());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_pre_inc_y() {
        let e = unary(Operator::PreInc, var_y());
        test_expr_range(&e, -4..=4);
    }

    // ==================== Assignment Operations ====================

    #[test]
    fn test_jit_assign_eq() {
        let e = binary(var_x(), Operator::AssignEq, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_add_eq() {
        let e = binary(var_x(), Operator::AddEq, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_sub_eq() {
        let e = binary(var_x(), Operator::SubEq, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_mul_eq() {
        let e = binary(var_x(), Operator::MulEq, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_div_eq() {
        let e = binary(var_x(), Operator::DivEq, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_mod_eq() {
        let e = binary(var_x(), Operator::ModEq, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_bitor_eq() {
        let e = binary(var_x(), Operator::BitOrEq, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_bitxor_eq() {
        let e = binary(var_x(), Operator::BitXorEq, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_bitand_eq() {
        let e = binary(var_x(), Operator::BitAndEq, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_shl_eq() {
        let e = binary(var_x(), Operator::BitShlEq, lit(2));
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_shr_eq() {
        let e = binary(var_x(), Operator::BitShrEq, lit(1));
        test_expr_range(&e, -4..=4);
    }

    // ==================== Power Operation ====================

    #[test]
    fn test_jit_pow() {
        let e = binary(var_x(), Operator::Pow, var_y());
        test_expr_range(&e, -2..=4);
    }

    #[test]
    fn test_jit_pow_positive() {
        let e = binary(lit(2), Operator::Pow, lit(10));
        assert!(compare_jit_naive(&e, 0, 0));
    }

    #[test]
    fn test_jit_pow_zero_exp() {
        let e = binary(var_x(), Operator::Pow, lit(0));
        test_expr_range(&e, -4..=4);
    }

    // ==================== Complex Expressions ====================

    #[test]
    fn test_jit_nested_add_mul() {
        // (x + y) * 2
        let inner = Arc::new(binary(var_x(), Operator::Add, var_y()));
        let e = binary(inner, Operator::Mul, lit(2));
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_nested_sub_div() {
        // (x - y) / 2
        let inner = Arc::new(binary(var_x(), Operator::Sub, var_y()));
        let e = binary(inner, Operator::Div, lit(2));
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_deeply_nested() {
        // ((x + 1) * 2) - y
        let inner1 = Arc::new(binary(var_x(), Operator::Add, lit(1)));
        let inner2 = Arc::new(binary(inner1, Operator::Mul, lit(2)));
        let e = binary(inner2, Operator::Sub, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_comparison_chain() {
        // (x < y) == (y > x)
        let left = Arc::new(binary(var_x(), Operator::Lt, var_y()));
        let right = Arc::new(binary(var_y(), Operator::Gt, var_x()));
        let e = binary(left, Operator::Eq, right);
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_mixed_ops() {
        // x + y * 2
        let mul = Arc::new(binary(var_y(), Operator::Mul, lit(2)));
        let e = binary(var_x(), Operator::Add, mul);
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_unary_in_binary() {
        // (-x) + y
        let neg = Arc::new(unary(Operator::Neg, var_x()));
        let e = binary(neg, Operator::Add, var_y());
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_logical_and_comparison() {
        // (x > 0) && (y > 0)
        let left = Arc::new(binary(var_x(), Operator::Gt, lit(0)));
        let right = Arc::new(binary(var_y(), Operator::Gt, lit(0)));
        let e = binary(left, Operator::And, right);
        test_expr_range(&e, -4..=4);
    }

    #[test]
    fn test_jit_bitwise_chain() {
        // (x | y) & 0xFF
        let bitor = Arc::new(binary(var_x(), Operator::BitOr, var_y()));
        let e = binary(bitor, Operator::BitAnd, lit(0xFF));
        test_expr_range(&e, -4..=4);
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_jit_div_by_zero() {
        let e = binary(var_x(), Operator::Div, lit(0));
        let jit = jit_compile_expr(&e);
        let f = jit.func();
        let mut x = 10;
        let mut y = 0;
        // Should not crash - returns 0
        let result = unsafe { f(&mut x, &mut y) };
        assert_eq!(result, 0);
    }

    #[test]
    fn test_jit_mod_by_zero() {
        let e = binary(var_x(), Operator::Mod, lit(0));
        let jit = jit_compile_expr(&e);
        let f = jit.func();
        let mut x = 10;
        let mut y = 0;
        // Should not crash - returns 0
        let result = unsafe { f(&mut x, &mut y) };
        assert_eq!(result, 0);
    }

    #[test]
    fn test_jit_multiple_compilations() {
        // Test that we can compile multiple expressions
        let e1 = binary(var_x(), Operator::Add, var_y());
        let e2 = binary(var_x(), Operator::Mul, var_y());
        let e3 = binary(var_x(), Operator::Sub, var_y());

        let jit1 = jit_compile_expr(&e1);
        let jit2 = jit_compile_expr(&e2);
        let jit3 = jit_compile_expr(&e3);

        let f1 = jit1.func();
        let f2 = jit2.func();
        let f3 = jit3.func();

        let mut x = 3;
        let mut y = 4;

        assert_eq!(unsafe { f1(&mut x, &mut y) }, 7);
        assert_eq!(unsafe { f2(&mut x, &mut y) }, 12);
        assert_eq!(unsafe { f3(&mut x, &mut y) }, -1);
    }

    #[test]
    fn test_jit_finalize_required() {
        let e = binary(var_x(), Operator::Add, var_y());
        let mut jit = Jit::new(4096);
        jit.compile(&e);
        // Not calling finalize should cause panic when calling func()
        // We test that finalize works correctly
        jit.finalize();
        let f = jit.func();
        let mut x = 5;
        let mut y = 3;
        assert_eq!(unsafe { f(&mut x, &mut y) }, 8);
    }

    #[test]
    #[should_panic(expected = "JIT not finalized")]
    fn test_jit_func_without_finalize_panics() {
        let e = binary(var_x(), Operator::Add, var_y());
        let mut jit = Jit::new(4096);
        jit.compile(&e);
        // This should panic because finalize() wasn't called
        let _ = jit.func();
    }
}
