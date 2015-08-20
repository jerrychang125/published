%% Jie Fu
% A Parallel Ant Colony Optimization Algorithm with GPU-Acceleration Based on All-In-Roulette Selection
% 2010 Third International Workshop on Advanced Computational Intelligence (IWACI), 
% requires Jacket GPU library

function MMASG(inputfile)
%% Example: MMASG('ulysses22.tsp')
%load cache (optional)
%S = mfilename('fullpath'); S = S(1:end-length(mfilename));
%gcache('load',[S 'MMASGcache.jkt']);
disp('AS is reading input nodes file...');
[Dimension,NodeCoord,NodeWeight,Name]=FileInput(inputfile);
disp([num2str(Dimension),' nodes in',Name,' has been read in']);
disp(['AS start at ',datestr(now)]);
%%%%%%%%%%%%% the key parameters of Ant System %%%%%%%%%
MaxITime=1000;
AntNum=Dimension;
alpha=1;
beta=2;
rho=0.02;
%%%%%%%%%%%%% the key parameters of Ant System %%%%%%%%%
[GBTour,GBLength,Option,IBRecord] = AS(NodeCoord,NodeWeight,AntNum,MaxITime,alpha,beta,rho);    
disp(['AS stop at ',datestr(now)]);
figure(1);
subplot(2,1,1)
plot(1:length(IBRecord(1,:)),IBRecord(1,:));
xlabel('Iterative Time');
ylabel('Iterative Best Cost');
title(['Iterative Course: ','GMinL=',num2str(GBLength),', FRIT=',num2str(Option.OptITime)]);
subplot(2,1,2)
plot(1:length(IBRecord(2,:)),IBRecord(2,:));
xlabel('Iterative Time');
ylabel('Average Node Branching');
figure(2);
DrawCity(NodeCoord,GBTour);
title([num2str(Dimension),' Nodes Tour Path of ',Name]);
%save cache (optional)
%S = mfilename('fullpath'); S = S(1:end-length(mfilename));
%gcache('save',[S 'MMASGcache.jkt']);
function [Dimension,NodeCoord,NodeWeight,Name]=FileInput(infile)
if ischar(infile)
    fid=fopen(infile,'r');
else
    disp('input file no exist');
    return;
end
if fid<0
    disp('error while open file');
    return;
end
NodeWeight = [];
while feof(fid)==0
    temps=fgetl(fid);
    if strcmp(temps,'')
        continue;
    elseif strncmpi('NAME',temps,4)
        k=findstr(temps,':');
        Name=temps(k+1:length(temps));
    elseif strncmpi('DIMENSION',temps,9)
        k=findstr(temps,':');
        d=temps(k+1:length(temps));
        Dimension=str2double(d); %str2num
    elseif strncmpi('EDGE_WEIGHT_SECTION',temps,19)
        formatstr = [];
        for i=1:Dimension
            formatstr = [formatstr,'%g '];
        end
        NodeWeight=fscanf(fid,formatstr,[Dimension,Dimension]);
        NodeWeight=NodeWeight';
    elseif strncmpi('NODE_COORD_SECTION',temps,18) || strncmpi('DISPLAY_DATA_SECTION',temps,20)
        NodeCoord=fscanf(fid,'%g %g %g',[3 Dimension]);
        NodeCoord=NodeCoord';
    end
end
fclose(fid);

function plothandle=DrawCity(CityList,Tours)
nc=length(Tours);
xd=zeros(1,nc);yd=zeros(1,nc);
plothandle=plot(CityList(:,2:3),'.');
set(plothandle,'MarkerSize',16);
for i=1:nc
    xd(i)=CityList(Tours(i),2);
    yd(i)=CityList(Tours(i),3);
end
set(plothandle,'XData',xd,'YData',yd);
line(xd,yd);

function [GBTour,GBLength,Option,IBRecord]=AS(CityMatrix,WeightMatrix,AntNum,MaxITime,alpha,beta,rho)
%% (Ant System) date:2010 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference：
% Dorigo M, Maniezzo Vittorio, Colorni Alberto. 
%   The Ant System: Optimization by a colony of cooperating agents [J]. 
%   IEEE Transactions on Systems, Man, and Cybernetics--Part B,1996, 26(1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global ASOption Problem AntSystem
ASOption = InitParameter(CityMatrix,AntNum,alpha,beta,rho,MaxITime);
Problem = InitProblem(CityMatrix,WeightMatrix);
AntSystem = InitAntSystem();
ITime=0;
IBRecord =zeros(3+ASOption.n,ASOption.MaxITime);
while 1
    InitStartPoint();
    P_ = CaculateShiftProb();
    for step = 2:ASOption.n
        Roulette(P_,step);
    end
    CloseTours();
    ITime = ITime + 1;
    CaculateToursLength();
    [IBL,IBT]=GlobleRefreshPheromone();
    ANB = CaculateANB();
    [GBTour,GBLength,IBRecord(:,ITime)] = GetResults(ITime,ANB,IBL,IBT);
%     ShowIterativeCourse(IBRecord(3:end,ITime),ITime,hline);
    if Terminate(ITime,ANB)
        break;
    end
end
Option = ASOption;
%% --------------------------------------------------------------
function ASOption = InitParameter(Nodes,AntNum,alpha,beta,rho,MaxITime)
ASOption.n = length(Nodes(:,1));
ASOption.m = AntNum;
ASOption.alpha_ = gsingle(alpha);
ASOption.beta_ = gsingle(beta);
ASOption.rho=rho;
ASOption.rho_ = gsingle(rho);
ASOption.MaxITime = MaxITime;
ASOption.OptITime = 1;
ASOption.Q_ = gsingle(10);
ASOption.C = 100;
ASOption.lambda_ = gsingle(0.1);
ASOption.ANBmin = 0; 
ASOption.GBLength = inf;
ASOption.GBTour = zeros(length(Nodes(:,1))+1,1);
ASOption.DispInterval = 10;
%% --------------------------------------------------------------
function Problem = InitProblem(Nodes,WeightMatrix)
global ASOption
n = length(Nodes(:,1));
Distances = WeightMatrix;
if isempty(WeightMatrix)
    Distances = CalculateDistance(Nodes);
end
%-------------convert over
Length=sum(diag(Distances([1:(n-1)], [2:n])))+Distances(n,1);
MaxTau=1/(ASOption.rho*Length);
root=nthroot(0.05,n);
rootkey=(1-root)/((n-3)*root);
MinTau=  MaxTau*rootkey;
MaxTau_=gsingle(MaxTau);
MinTau_=gsingle(MinTau);
%MaxTau_/(two_*n_);  just for testing
MatrixTau_=(gones(n)-geye(n))*MaxTau;
Distances_=gsingle(Distances);
Distances_=round(Distances_);
Problem = struct('nodes',Nodes,'dis_',Distances_,'tau_',MatrixTau_,'maxtau_',MaxTau_,'mintau_',MinTau_,'rootkey',rootkey);
%% --------------------------------------------------------------
function AntSystem = InitAntSystem()
global ASOption
n=ASOption.n;
m=ASOption.m;
% Penalty矩阵初始为1，如果走过了那座城市，那个点的Penalty为0
AntToursMatrix = zeros(m,n+1); 
ToursLength_ = gzeros(m,1);  
PenaltyMatrix_=gones(m, n); 
%---------------------
AntSystem=struct('tours',AntToursMatrix,'lengths_',ToursLength_,'penalty_',PenaltyMatrix_);
%% --------------------------------------------------------------
function InitStartPoint()
global AntSystem ASOption
m=ASOption.m;     
n=ASOption.n;
AntSystem.tours(:,1)=ceil( ( rand(m,1) ).*m );%经测试，成功
%将初始随机位置的Penalty设置为0
IndexVector=AntSystem.tours(:, 1);
AntSystem.penalty_=gones(m,n);
for k=1:n
        AntSystem.penalty_(k,IndexVector(k,1))=0;
end
AntSystem.lengths_ = gzeros(m,1); %经测试，成功
%% --------------------------------------------------------------
function Probs_ = CaculateShiftProb()
%一次性把Probs算出来，所有蚂蚁的Probs都是一样的
%不同的路径选择源于后面的Roulette，尤其是里面的随机数
%经测试，可行
global ASOption Problem
n=ASOption.n;
Probs_=gzeros(n);
one_=gsingle(1);
gfor k=1:n
       Probs_(k, :)=(Problem.tau_(k, :).^ASOption.alpha_).*((one_./Problem.dis_(k, :)).^ASOption.beta_);
       temp_=sum(Probs_);  
       temp_=repmat(temp_,n,1);
       Probs_=Probs_ ./temp_;  
gend
%% --------------------------------------------------------------
function Roulette(P_,s)
%一次性算出概率矩阵，迭代n次，每走一步运算一次，然后更新PenaltyMatrix
%penalty使得走过的node为0，以使得Select值最低，从而不被选择
global ASOption AntSystem
m=ASOption.m;
n=ASOption.n;
Dice_=grand(m, n);  
SelectVector =AntSystem.tours(:, (s-1));
SelectProbsMatrix_=P_(SelectVector,:);
NextNodeCandidates_=AntSystem.penalty_.*Dice_.*SelectProbsMatrix_;
[DumpValue, Index_]=max(NextNodeCandidates_,[],2);
Index=single(Index_);
[Dump,ColumnSub]=ind2sub([m,n],Index);%ind2sub runs faster on CPU
AntSystem.tours(:, s)=ColumnSub;
penalty=single(AntSystem.penalty_);
penalty(Index)=0;
AntSystem.penalty_=gsingle(penalty);
%% --------------------------------------------------------------
function CloseTours()
global AntSystem ASOption
n=ASOption.n;
AntSystem.tours(:,n+1) = AntSystem.tours(:,1);
%% --------------------------------------------------------------
function CaculateToursLength()
global AntSystem ASOption Problem
n=ASOption.n;
m=ASOption.m;
IndexMatrix1=AntSystem.tours(:, [1:n]);
IndexMatrix2=AntSystem.tours(:, [2:(n+1)]);
Problem.dis=single(Problem.dis_);
RouteDistanceMatrix=zeros(m,n);
%for-loop runs faster on CPU
for i=1:m
    RouteDistanceMatrix(i,:)=diag(Problem.dis(IndexMatrix1(i,:),IndexMatrix2(i,:)));
end
RouteDistanceMatrix_=gsingle(RouteDistanceMatrix);
gfor k=1:m
       AntSystem.lengths_(k)=sum (RouteDistanceMatrix_(k,:) );
gend
%% --------------------------------------------------------------
function [GBTour ,GBLength ,Record ] = GetResults(ITime,ANB,IBLength,IBTour)
global ASOption
if IBLength<=ASOption.GBLength 
  ASOption.GBLength = IBLength;
	ASOption.GBTour = IBTour;
	ASOption.OptITime = ITime;
end
GBTour = ASOption.GBTour';
GBLength = ASOption.GBLength;
Record = [IBLength,ANB,IBTour]';
%% --------------------------------------------------------------
function [IBLength,IBTour]=GlobleRefreshPheromone()
global AntSystem ASOption Problem
n=ASOption.n;
one_=gsingle(1);
[IBLength_ ,AntIndex_ ] = min(AntSystem.lengths_);
AntIndex=single(AntIndex_);
IBLength=single(IBLength_);
IBTour = AntSystem.tours(AntIndex,:);
IndexVector1=IBTour(1:n);
IndexVector2=IBTour(2:(n+1));
sumdtau=zeros(n,n);
sumdtau(sub2ind([n,n],IndexVector1,IndexVector2))=1/IBLength;
sumdtau_=gsingle(sumdtau);
maxtau=1/(ASOption.rho*IBLength);
mintau=maxtau*Problem.rootkey;
Problem.maxtau_=gsingle(maxtau);
Problem.mintau_=gsingle(mintau);
Problem.tau_ =Problem.tau_*(one_-ASOption.rho_)+sumdtau_ ;
%一次性更新完毕
Problem.tau_((Problem.tau_<Problem.mintau_) &(Problem.tau_~=0))=Problem.mintau_;
Problem.tau_(Problem.tau_>Problem.maxtau_)=Problem.maxtau_;
%% --------------------------------------------------------------
function flag = Terminate(ITime,ANB)
global ASOption
flag = false;
if ANB<=ASOption.ANBmin || ITime>=ASOption.MaxITime
    flag = true;
end
%% --------------------------------------------------------------
function ANB_ = CaculateANB()
global ASOption Problem
n=ASOption.n;
n_=gsingle(n);
two_=gsingle(2);
NB_=sum(Problem.tau_~=Problem.mintau_)-gones(1,n)*two_;
ANB_=sum(NB_)/n_;
ANB=single(ANB_);
if ANB<0.01
   Problem.tau_=(gones(n)-geye(n))*Problem.maxtau_;
end
%% --------------------------------------------------------------
function Distances = CalculateDistance(Nodes)
global ASOption 
Nodes(:,1)=[]; 
Distances=inf(ASOption.n,ASOption.n);
for i=2:ASOption.n
    for j=1:i
        if(i==j)    
            continue;
        else
            dij=Nodes(i,:)-Nodes(j,:);
            Distances(i,j)=sqrt(dij(1)^2+dij(2)^2);
            Distances(j,i)=Distances(i,j);  
        end
    end
end