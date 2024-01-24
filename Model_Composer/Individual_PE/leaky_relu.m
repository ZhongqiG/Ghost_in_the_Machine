function y = leaky_relu(x)

a = fi(0.1, 1, 16, 3);
if x> a*x
    y = x;
else
    y = a*x;


end





