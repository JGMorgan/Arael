var mlmath = require('./ml_math.js');
var LinearRegression = require('./linear_regression.js');
class LogisticRegression extends LinearRegression.LinearRegression{
    predict(X){
        return mlmath.sigmoid(super.predict(X));
    }
}
module.exports = {LogisticRegression};
