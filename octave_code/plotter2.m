close all; clear all;
load('data/run2.mat'); 


figure(1);
tit = title('SVDs');
hold on;
plot(diag(S),'r','linewidth',3);
hold off;
xl = xlabel('k');
yl = ylabel('sigma_k');
set(gca,'fontsize',20);
set(tit,'fontsize',20);
set(xl,'fontsize',20);
set(yl,'fontsize',20);
print('-depsc','images/svds2.eps');


figure(2);
tit = title('LOG ERRORS');
hold on;
plot(ks,log(0.01*percent_errors1),'r-*','linewidth',3);
plot(ks,log(0.01*percent_errors2),'g-*','linewidth',3);
plot(ks,log(0.01*percent_errors3),'b-*','linewidth',3);
hold off;
leg = legend('V1','V2','V3');
xl = xlabel('k');
yl = ylabel('log ||A - Uk Sk Vkt||_2/||A||_2');
set(gca,'fontsize',20);
set(tit,'fontsize',20);
set(xl,'fontsize',20);
set(yl,'fontsize',20);
set(leg,'fontsize',20);
print('-depsc','images/perrors2.eps');

