%ridge regression core code
% Initialize variables for regression analysis
x0 = (1:long)'; % Time index vector
x1 = [ones(long, 1), x0]; % Design matrix with intercept

for i = 1:m*n
    % Extract data for current iteration
    NEP = NEP_sum(i, :)'; 
    DFMI = DFMI_sum(i, :)'; 
    SM = SM_sum(i, :)'; 
    CEB = CEB_sum(i, :)'; 
    PET = PET_sum(i, :)'; 
    VPD = VPD_sum(i, :)'; 

    % Check if all variables are valid (no NaNs) and NEP has variation
    if all(~isnan([NEP, DFMI, SM, CEB, PET, VPD])) && numel(unique(NEP)) > 1
        % Prepare data for ridge regression
        x = normalize([DFMI, CEB, SM, PET, VPD], "range"); % Normalize predictors
        y = normalize(NEP, "range"); % Normalize response
        k = 1:0.1:10; % Range of lambda values for ridge regression
        
        % Perform ridge regression
        B = ridge(y, x, k, 0); 
        error = nan(length(k), 1); % Initialize error storage

        % Calculate errors for each lambda
        for k1 = 1:length(k)
            A = B(:, k1); % Coefficients for current lambda
            yn = A(1) + x * A(2:end); % Predicted values
            err = abs(y - yn) ./ y; % Relative error
            error(k1) = sum(err(~isinf(err))) / length(y(y > 0)); % Average error
        end

        % Find lambda with minimum error
        index = find(error == min(error));
        ridge_r = ridge(y, x, k(index(1)), 0); % Final ridge coefficients
        
        % Calculate trends for each predictor and response
        trends = arrayfun(@(j) regress(x(:, j), x1), 1:size(x, 2)); % Trends for predictors
        y_trend = regress(y, x1); % Trend for response
        yn = [ones(long, 1), x] * ridge_r; % Predicted values from ridge regression
        yn_trend = regress(yn, x1); % Trend for predicted values

        % Calculate normalized contributions
        nc = abs(ridge_r(2:end) .* trends);
        nrc = (nc / sum(nc)) * 100; % Normalized relative contributions

    else
        % Assign NaN values if data is insufficient
        ridge_r = nan(1, 6);
        nrc = nan(1, 5);
    end

    % Store results for current iteration
    DFMI_r(i) = ridge_r(2);
    CEB_r(i) = ridge_r(3);
    SM_r(i) = ridge_r(4);
    PET_r(i) = ridge_r(5);
    VPD_r(i) = ridge_r(6);
    
    DFMI_nrc(i) = nrc(1);
    CEB_nrc(i) = nrc(2);
    SM_nrc(i) = nrc(3);
    PET_nrc(i) = nrc(4);
    VPD_nrc(i) = nrc(5);
end
