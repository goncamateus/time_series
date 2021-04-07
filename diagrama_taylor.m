clc, close all, clear all;


% carregando os dados
pred_attD = load('data/out/KIATAWF1/KIATAWF1_Nosso_decomp.mat');
pred_attD = double(pred_attD.KIATAWF1);
pred_att = load('data/out/KIATAWF1/KIATAWF1_Nosso_pure.mat');
pred_att = double(pred_att.KIATAWF1);
pred_mlpD = load('data/out/KIATAWF1/KIATAWF1_MLP_decomp.mat');
pred_mlpD = double(pred_mlpD.KIATAWF1);
pred_mlp = load('data/out/KIATAWF1/KIATAWF1_MLP_pure.mat');
pred_mlp = double(pred_mlp.KIATAWF1);


target = load('data/out/KIATAWF1/KIATAWF1_ref.mat');
target = double(target.KIATAWF1);

Nregressoras = 3;  
timeSteps = 12;

for j=1:timeSteps
    
    %obs = target(Nregressoras+j:end,1);
    obs = target(:,j);
    stdObs = nanstd(obs);   
    
    
    %-------------------- nosso decomp ------------------------------

    previsao = pred_attD(:,j);
    %previsao = previsao(1:end-(Nregressoras+j)+1,1);
        
    %obsPy = obsPy(~isnan(previsao));
    %previsao = previsao(~isnan(obsPy2));
    %obsPy = obsPy(~isnan(obsPy));
    %previsao = previsao(~isnan(previsao));
    
    correlacaoATTD(j,1) = corr(previsao,obs);
    stdPrevATTD(j,1) = nanstd(previsao);
    rmsdATTD(j,1) = sqrt(stdObs.^2 + stdPrevATTD(j,1)^2 - 2*stdPrevATTD(j,1)*stdObs.*correlacaoATTD(j,1));
    
    
    %-------------------- nosso pure ------------------------------

    previsao = pred_att(:,j);
    %previsao = previsao(1:end-(Nregressoras+j)+1,1);
        
    %obsPy = obsPy(~isnan(previsao));
    %previsao = previsao(~isnan(obsPy2));
    %obsPy = obsPy(~isnan(obsPy));
    %previsao = previsao(~isnan(previsao));
    
    correlacaoATT(j,1) = corr(previsao,obs);
    stdPrevATT(j,1) = nanstd(previsao);
    rmsdATT(j,1) = sqrt(stdObs.^2 + stdPrevATT(j,1)^2 - 2*stdPrevATT(j,1)*stdObs.*correlacaoATT(j,1));
  
        %-------------------- MLP pure ------------------------------

    previsao = pred_mlp(:,j);
    %previsao = previsao(1:end-(Nregressoras+j)+1,1);
        
    %obsPy = obsPy(~isnan(previsao));
    %previsao = previsao(~isnan(obsPy2));
    %obsPy = obsPy(~isnan(obsPy));
    %previsao = previsao(~isnan(previsao));
    
    correlacaoMLP(j,1) = corr(previsao,obs);
    stdPrevMLP(j,1) = nanstd(previsao);
    rmsdMLP(j,1) = sqrt(stdObs.^2 + stdPrevMLP(j,1)^2 - 2*stdPrevMLP(j,1)*stdObs.*correlacaoMLP(j,1));
    
    %-------------------- MLP decomp ------------------------------

    previsao = pred_mlpD(:,j);
    %previsao = previsao(1:end-(Nregressoras+j)+1,1);
        
    %obsPy = obsPy(~isnan(previsao));
    %previsao = previsao(~isnan(obsPy2));
    %obsPy = obsPy(~isnan(obsPy));
    %previsao = previsao(~isnan(previsao));
    
    correlacaoMLPD(j,1) = corr(previsao,obs);
    stdPrevMLPD(j,1) = nanstd(previsao);
    rmsdMLPD(j,1) = sqrt(stdObs.^2 + stdPrevMLPD(j,1)^2 - 2*stdPrevMLPD(j,1)*stdObs.*correlacaoMLPD(j,1));
    
end


desvios = [stdObs;stdPrevATTD;stdPrevMLPD];
erros = [0;rmsdATTD;rmsdMLPD];
correlacoes = [1;correlacaoATTD;correlacaoMLPD];

PP = {'O' 'att' 'mlp'}; 
COLORS = {'r' 'b' 'm'};
TD(desvios,erros,correlacoes,PP,COLORS,1,0);

% desvios = [stdObs;stdPrevATT;stdPrevMLP];
% erros = [0;rmsdATT;rmsdMLP];
% correlacoes = [1;correlacaoATT;correlacaoMLP];
% 
% PP = {'O' 'att' 'mlp'}; 
% COLORS = {'r' 'b' 'm'};
% TD(desvios,erros,correlacoes,PP,COLORS,1,0);
cd 'data/out/'
