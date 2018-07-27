from pylab import *
x=linspace(-5*pi,5*pi,500)
plot(x,x,'b')
plot(x,-x,'g')
plot(x, sin(x),'b',linewidth=3)
plot(x,x*sin(x),'r',linewidth=3)
legend(['sinx'])
xlabel('x')










ylabel('sinx')
show()
