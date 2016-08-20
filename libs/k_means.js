var mlmath = require('./ml_math.js');
class KMeans {
    constructor(K){
        this.K = K;
    }

    cluster(X){
        var prev = [];
        var out = [1];
        var centroids = [];
        var minXVals = X[0].slice();
        var maxXVals = X[0].slice();
        for (var i = 1; i < X.length; i++){
            for (var j = 0; j < X[i].length; j++){
                if (minXVals[j] > X[i][j]){
                    minXVals[j] = X[i][j];
                }
                if (maxXVals[j] < X[i][j]){
                    maxXVals[j] = X[i][j];
                }
            }
        }
        for (var i = 0; i < this.K; i++){
            prev.push([]);

            var temp = [];
            for (var j = 0; j < X[0].length; j++){
                temp.push(Math.floor((Math.random() * maxXVals[j]) + minXVals[j]));
            }
            centroids.push(temp);
        }
        while (JSON.stringify(prev) !== JSON.stringify(centroids)){
            out = [];
            prev = [];
            for (var i = 0; i < this.K; i++){
                out.push([]);
            }
            for (var i = 0; i < X.length; i++){
                var min = mlmath.distance(X[i], centroids[0]);
                var minIndex = 0;
                for (var j = 1; j < this.K; j++){
                    var distance = mlmath.distance(X[i], centroids[j]);
                    if (distance < min){
                        min = distance;
                        minIndex = j;
                    }
                }
                out[minIndex].push(X[i].slice());
            }
            prev = mlmath.clone(centroids);
            for (var i = 0; i < this.K; i++){
                if (out[i].length === 0){
                    continue;
                }
                for (var j = 0; j < centroids[0].length; j++){
                    var sum = 0;
                    for (var k = 0; k < out[i].length; k++){
                        sum += out[i][k][j];
                    }
                    centroids[i][j] = sum/out[i].length;
                }
            }
        }
        return out;
    }
}
module.exports = {KMeans};
