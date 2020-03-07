%% Implementa uma rede neuronal 
%  com 4 entradas, uma camada oculta e 4 saidas
%
clear variables
% Para o treinamento:
%      L0                     L1                   L2
% +------------+  Syn0  +-------------+  Syn1  +---------+ 
% | 4 entradas |------->| 32 neuronios|------->|4 Saidas |
% |   16x4     |  4x32  |    16x32    |  32x4  |  16x4   |
% +------------+        +-------------+        +---------+
% 16 registros de treino
% com 4 entradas cada
%
% Entradas de treinamento e ...
X = [-1,-1,-1,-1;
     -1,-1,-1, 1;
     -1,-1, 1,-1;
     -1,-1, 1, 1;
     -1, 1,-1,-1;
     -1, 1,-1, 1;
     -1, 1, 1,-1;
     -1, 1, 1, 1;
      1,-1,-1,-1;
      1,-1,-1, 1;
      1,-1, 1,-1;
      1,-1, 1, 1;
      1, 1,-1,-1;
      1, 1,-1, 1;
      1, 1, 1,-1;
      1, 1, 1, 1];
% ... suas respectivas saidas:
y = [1,0,0,0;
     1,0,0,0;
     1,0,0,0;
     1,0,0,0;
     0,1,0,0;
     0,1,0,0;
     0,1,0,0;
     0,1,0,0;
     0,0,1,0;
     0,0,1,0;
     0,0,1,0;
     0,0,1,0;
     0,0,0,1;
     0,0,0,1;
     0,0,0,1;
     0,0,0,1];

alpha = 0.15; % Tamanho do passo
HLS = 32; % Tamanho da camanda oculta
Precisao = 1E-4;
% Tamanhos: 
[L,C]=size(X); % L linhas (registros) X C colunas (entradas)
[~,S]=size(y); % Qtde de saidas: S
%
% Inicializa os pesos com numeros aleatorios entre -1 e 1
% As sinapses 0 conectam a camada 0 (entrada) com a camada 1 (c. oculta)
Syn0 = 2*rand(C,HLS)-1; 
% As sinapses 1 conectam a camada 1 com a camada 2 (saida)
Syn1 = 2*rand(HLS,C)-1; 
%
% Inicializa as camadas da rede e suas variaveis auxiliares:
L0                  = X;             
L1(1:L,1:HLS)       = 0;  
L2(1:L,1:S)         = 0;    
L1_delta(1:L,1:HLS) = 0;
L2_delta(1:L,1:S)   = 0;
%
tic;
for I=1:300000 % treinamento em varias iteracoes
    %% Calcula a camada 1:
    Prod = L0 * Syn0; % Poduto matricial L0 X Syn0
    %Passa o resultado na Sigmoid para obter a camada 1:
    %for J=1:L
    %    for K=1:HLS
    %        L1(J,K) = Sigmoid(Prod(J,K),0);
    %    end
    %end
    L1 = Sigmoid(Prod,0);
    
    %% Calcula a camada 2:
    Prod = L1 * Syn1; % Poduto matricial L1 X Syn1
    %Passa o resultado na Sigmoid para obter a camada 2:
    %for J=1:L
    %    for K=1:S
    %        L2(J,K) = Sigmoid(Prod(J,K),0);
    %    end
    %end
    L2 = Sigmoid(Prod,0);
    
    % A partir daqui - backpropagation:
    %% Calcula os novos pesos para as sinapses 1:
    L2_loss = y - L2; % Funcao perda para a camada 2
    MSE = sum(sum(L2_loss .^ 2)/L); % Mean Squared Error
    % Se o MSE ja esta aceitavel, encerra as iteracoes:
    if MSE<Precisao
        break
    end
    % A cada 1000 iteracoes, mostra o MSE:
    if mod(I,1000)==0 
        disp(MSE);
    end
    %
    %for J=1:L
    %    for K=1:S
    %        % Multiplica as perdas pelos gradientes da sigmoid:
    %        L2_delta(J,K) = L2_loss(J,K) * Sigmoid(L2(J,K),1);
    %    end
    %end
    L2_delta = L2_loss .* Sigmoid(L2,1);
    
    % Atualiza os pesos das sinapses 1:
    Syn1 = Syn1 + alpha*(transpose(L1) * L2_delta); 
    
    %% Calcula os novos pesos para as sinapses 0:
    L1_loss = L2_delta * transpose(Syn1); % Funcao perda para a camada 1
    %for J=1:L
    %    for K=1:HLS
    %        % Multiplica as perdas pelos gradientes da sigmoid:
    %       L1_delta(J,K)=L1_loss(J,K)*Sigmoid(L1(J,K),1);
    %    end
    %end
    L1_delta =  L1_loss .* Sigmoid(L1,1);
    
    % Atualiza os pesos das sinapses 0:
    Syn0 = Syn0 + alpha*(transpose(L0) * L1_delta); 
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Testa a rede:
L0 =  [1, -1, 1, -1]; % Entrada
Prod = L0 * Syn0; % Produto matricial
clear L1;
%L1(1:S,1:HLS)=0;
%Passa o resultado na Sigmoid para obter a camada 1:
%for J=1:1
%    for K=1:HLS
%        L1(J,K) = Sigmoid(Prod(J,K),0);
%    end
%end
L1 = Sigmoid(Prod,0);
Prod = L1 * Syn1;

clear L2
%L2(1,1:S)=0;
%Passa o resultado na Sigmoid para obter a saida:
%for K=1:S
%    L2(1,K) = Sigmoid(Prod(1,K),0);
%end
L2 = Sigmoid(Prod,0);

disp(round(L2));
toc;