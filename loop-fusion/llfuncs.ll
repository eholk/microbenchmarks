
declare double    @llvm.sin.f64(double %Val)

declare double    @llvm.cos.f64(double %Val)

define double @ll_sin(double %x) readnone nounwind alwaysinline {
  %r = tail call double @llvm.sin.f64(double %x)
  ret double %r
}

define double @ll_cos(double %x) readnone nounwind alwaysinline {
  %r = tail call double @llvm.cos.f64(double %x)
  ret double %r
}
