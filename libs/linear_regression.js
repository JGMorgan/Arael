var mlmath = require('./ml_math.js');
class LinearRegression {
    constructor(num_features){
        this.Theta = [];
        for (var i = 0; i < num_features + 1; i++){
            this.Theta.push([0]);
        }
    }

    predict(X){
        if(this.Theta.length !== X.length){
            X.push(1);
        }
        var sum = 0;
        for (var i = 0; i < X.length; i++){
            sum += this.Theta[i][0] * X[i];
        }
        return sum;
    }

    train(X, Y, alpha){
        //((X^T * X)^-1) * X^T * Y
        for (var i = 0; i < X.length; i++){
            X[i].push(1);
        }
        if (alpha === undefined){
            alpha = 0.03;
        }
        var trX = mlmath.transpose(X);
        var d = mlmath.determinant(mlmath.dot(trX, X));
        if (d !== 0){
            this.Theta = mlmath.dot(mlmath.dot(mlmath.inverse(mlmath.dot(trX, X)), trX), Y);
        }else{
            var prev = [];
            for (var i = 0; i < this.Theta.length; i++){
                prev.push([1]);
            }
            while (JSON.stringify(prev) !== JSON.stringify(this.Theta)){
                prev = mlmath.clone(this.Theta);
                for (var i = 0; i < this.Theta.length; i++){
                    var sum = 0;

                    for (var j = 0; j < X.length; j++){
                        sum += (Y[j][0] - this.predict(X[j])) * X[j][i];
                    }
                    this.Theta[i][0] = this.Theta[i][0] + (alpha * sum);
                }
            }
        }
    }
}
module.exports = {LinearRegression};
