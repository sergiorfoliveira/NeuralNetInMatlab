function [s] = Sigmoid(x, flag)
    s = 1 ./ (1+2.718281828459045.^(-x));
    if(flag==1)
        s = s .* (1-s); % Retorna o valor da derivada no ponto x
   % else
   %      % Retorna o valor da funcao no ponto x
    end
end