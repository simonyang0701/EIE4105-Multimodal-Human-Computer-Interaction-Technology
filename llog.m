function l = llog(x)
x(x<2.2251e-308) = 2.2251e-308;
l = log(x);
