%% EXemplo 1
x = 1:8;
y = [1, 1.96, 2.93, 3.91, 4.57, 5.48, 6.38, 7.29];
y2 = [1, 1.96, 1.64, 2.42, 2.47,3.11, 3.30, 3.86];
bars = bar(x,[y2;y]);
grid()
xlabel('Number of threads');
ylabel('Speedup ');
title('Relationship between thread count and speedup increase');
legend("Without optimization","With optimization")

xtips1 = bars(1).XEndPoints;
ytips1 = bars(1).YEndPoints;
labels1 = string(bars(1).YData + "x");
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips2 = bars(2).XEndPoints;
ytips2 = bars(2).YEndPoints;
labels2 = string(bars(2).YData + "x");
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

%% Exemplo 2

x = 1:64;
y = [4.79, 9.42, 9.42, 11.47, 12.13, 15.48, 16.60, 18.84, 21.22, 23.44, 24.41, 25.54 ...
    26.95, 28.81, 29.39, 29.72, 30.43, 30.72, 30.74, 30.74 ,30.74 , 30.74 , 30.74 , 30.74 , 30.74 , 30.74...
30.74,30.74,30.74,30.74,30.74,30.74,30.74,30.74, 31.16 , 31.16 ,31.16, 31.18, 31.18, 31.18, 31.18, 31.18, ...
31.18, 31.18 , 31.18, 31.18 , 31.18, 31.23, 31.29, 31.31, 31.42, 31.42 , 31.42, 31.42 , 31.42 , ...
31.42 , 31.42, 31.50, 31.50 ,31.50,31.50,31.50,31.50,31.50];

% plot(x,y, '-', 'linewidth', 3)
% hold on
% plot(x,y,'.','MarkerSize',20)
bars = bar(x(2:2:64),y(2:2:64), 0.8);
grid()
xtips1 = bars(1).XEndPoints;
ytips1 = bars(1).YEndPoints;
labels1 = string(bars(1).YData);
labels2 = string(bars(1).XData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
text(xtips1,ytips1,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','cap')
xlabel('Number of tasks');
ylabel('Speedup');
title('Relationship between the number of tasks and speedup increase');

%% Este trabalho 
close all
clear 
clc
% M=6 => SP = 8.075
% M = 10 => SP = 8.368
% M = 20 => SP = 8.620
% M = 60 => SP = 8.756
% M = 100 => SP = 8.816
% M = 150 => SP = 8.827

x2 = [6, 10, 20, 60, 100, 150];
x = 1:6;
y = [8.075, 8.368, 8.620, 8.756, 8.816, 8.827];

% plot(x,y, '-', 'linewidth', 3)
% hold on
% plot(x,y,'.','MarkerSize',20)
bars = bar(x,y);
grid()
xtips1 = bars(1).XEndPoints;
ytips1 = bars(1).YEndPoints;
labels1 = string(bars(1).YData);
% labels2 = string(x2(bars(1).XData));
set(gca, 'XTickLabel', x2);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
% text(xtips1,ytips1,labels2,'HorizontalAlignment','center',...
%     'VerticalAlignment','cap')
xlabel('M');
ylabel('Speedup');
title('Speedup of BenchMax as a Function of M (N=20)');
ylim([0 10]);
