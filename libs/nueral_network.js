var mlmath = require('./ml_math.js');

class NeuralNetwork{
    /**
     * @param numHiddenLayers is the number of hidden layers in the network
     * @param neuronsPerLayer is an array of ints, each int corresponds to
     * number of neurons in that hidden layer
     * */
    constructor(numFeatures, numHiddenLayers, neuronsPerLayer, activation){
        this.numHiddenLayers = numHiddenLayers;
        this.neuronsPerLayer = neuronsPerLayer;
        if (activation !== undefined){
            this.activation = activation;
        }else{
            this.activation = mlmath.sigmoid;
        }
        this.Weights = [];
        var temp = [];
        for (var i = 0; i < numFeatures; i++){
            var row = [];
            for (var j = 0; j < neuronsPerLayer[0]; j++){
                row.push(Math.random());
            }
            temp.push(row);
        }
        this.Weights.push(temp);
        for (var i = 1; i < numHiddenLayers; i++){
            temp = [];
            for (var j = 0; j < neuronsPerLayer[i-1]; j++){
                var row = [];
                for (var k = 0; k < neuronsPerLayer[i]; k++){
                    row.push(Math.random());
                }
                temp.push(row);
            }
            this.Weights.push(temp);
        }
        temp = [];
        for (var j = 0; j < neuronsPerLayer[0]; j++){
            temp.push([Math.random()]);
        }
        this.Weights.push(temp);
    }

    get weights(){
        return this.Weights;
    }

    set weights(weights){
        this.Weights = weights;
    }

    predict(X){
        this.Z = [];
        var out = mlmath.clone(X);
        for (var i = 0; i < this.numHiddenLayers + 1; i++){/*
            console.log("OUT");
            console.log(out);
            console.log("WEIGHTS");
            console.log(this.Weights[i]);*/
            out = mlmath.dot(mlmath.clone(out), this.Weights[i]);
            this.Z.push(mlmath.clone(out));
            out = out.map((x) => x.map((y) => this.activation(y)));
        }
        return out;
    }

    cost(X, Y){
        var errorSum = 0;
        for (var i = 0; i < Y.length; i++){
            var yHat = this.predict(X[i]);
            errorSum += Math.pow(Y[i][0] - yHat[0][0], 2);
        }
        return errorSum / 2;
    }

    costPrime(X, Y){
        var clone = (x) => {
            var out = [];
            for (var i = 0; i < x.length; i++){
                var inner = [];
                for (var j = 0; j < x[i].length; j++){
                    inner.push(x[i][j].slice(0));
                }
                out.push(inner);
            }
            return out;
        }
        var activate = (Z) => Z.map((x) => x.map((y) => this.activation(y)));
        var flip = (arr) => {
            var out = [];
            for (var i = 0; i < arr.length; i++){
                out.push(arr[arr.length - i - 1]);
            }
            return out;
        };
        var yHat = this.predict(X);
        var Z = clone(this.Z);
        var Weights = clone(this.Weights);
        var DJDW = [];
        var delta = [[-(Y[0] - yHat[0][0]) * mlmath.sigmoidPrime(Z[Z.length - 1][0][0])]];
        DJDW.push(mlmath.dot(mlmath.transpose(activate(Z[Z.length - 2])), delta));

        for (var i = 0; i < Weights.length - 2; i++){
            delta = mlmath.dot(delta, mlmath.transpose(Weights[Weights.length - i - 1]));
            for (var j = 0; j < Z[i][0].length; j++){
                delta[0][j] = delta[0][j] * Z[i][0][j];
            }
            DJDW.push(mlmath.dot(mlmath.transpose(activate(Z[Z.length - i - 3])), delta));
        }
        delta = mlmath.dot(delta, mlmath.transpose(Weights[0]));
        for (var i = 0; i < Z[0][0].length; i++){
            delta[0][i] = delta[0][i] * Z[0][0][i];
        }
        DJDW.push(mlmath.dot(mlmath.transpose(X), delta));
        DJDW = flip(DJDW);
        return DJDW;
    }

    /**
     * @param X is the input vector
     * @param Y is the vector of expected results
     * @param alpha is the learning rate
     * @param train_percent what percentage of the training data is used for training
     * trains the neural network
     * */
    train(X, Y, alpha, train_percent){
        for (var i = 0; i < Math.floor(train_percent * X.length); i++){
            var djdw = this.costPrime(X[i], Y[i]);
            for (var j = 0; j < djdw.length; j++){
                for (var k = 0; k < djdw[j].length; k++){
                    for (var l = 0; l < djdw[j][k].length; l++){
                        this.Weights[j][k][l] = this.Weights[j][k][l] - (alpha * djdw[j][k][l]);
                    }
                }
            }
        }
    }
}

var nn = new NeuralNetwork(3, 7, [3,3,3,3,3,3,3]);
console.log("prediction   " + nn.predict([[5,2,3]]));
console.log(nn.cost([[[5,2,3]],[[5,2,3]],[[5,2,3]],[[5,2,3]]], [[.9],[.9],[.9],[.9]]));
for (var i = 0; i < 15; i++){
    nn.train([[[5,2,3]],[[5,2,3]],[[5,2,3]],[[5,2,3]]], [[.9],[.9],[.9],[.9]], 0.01, 1);
}
console.log("prediction   " + nn.predict([[5,2,3]]));
console.log(nn.cost([[[5,2,3]],[[5,2,3]],[[5,2,3]],[[5,2,3]]], [[.9],[.9],[.9],[.9]]));
var nn2 = new NeuralNetwork(3, 7, [3,3,3,3,3,3,3]);
console.log("prediction   " + nn2.predict([[5,2,3]]));
console.log(nn2.cost([[[5,2,3]],[[5,2,3]],[[5,2,3]],[[5,2,3]]], [[.9],[.9],[.9],[.9]]));
for (var i = 0; i < 5; i++){
    nn2.train([[[5,2,3]],[[5,2,3]],[[5,2,3]],[[5,2,3]]], [[.9],[.9],[.9],[.9]], 0.01, 1);
}
console.log("prediction   " + nn2.predict([[5,2,3]]));
console.log(nn2.cost([[[5,2,3]],[[5,2,3]],[[5,2,3]],[[5,2,3]]], [[.9],[.9],[.9],[.9]]));
