function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.001;
sigma = 0.001;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


C /= 3;
sigma /= 3;
result = [];

for i = 1:20
  C *= 3;
  for j = 1:20
    sigma *= 3;
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    yPredict = svmPredict(model, Xval);
    error = mean(double(yval ~= yPredict));
    tmpResult = [C sigma error];
    result = [result; tmpResult];
  endfor
endfor

[num, index] = min(result);
C = result(index(3), 1);
sigma = result(index(3), 2);
disp(sprintf("C=%f", C));
disp(sprintf("sigma=%f", sigma));
disp(sprintf("error=%f", num(3)));
%disp(result);





% =========================================================================

end
