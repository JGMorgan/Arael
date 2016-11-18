var transpose = (x) => {
    var out = [];
    for (var i = 0; i < x[0].length; i++){
        var temp = [];
        for (var j = 0; j < x.length; j++){
            temp.push(x[j][i]);
        }
        out.push(temp);
    }
    return out;
}

var clone = (x) => {
    var out = [];
    for (var i = 0; i < x.length; i++){
        out.push(x[i].slice(0));
    }
    return out;
}

var determinant = (x) => {
    if(x.length === 2){
        return x[0][0] * x[1][1] - x[1][0] * x[0][1];
    }else if(x.length === 3){
        return x[0][0] * (x[1][1] * x[2][2] - x[1][2] * x[2][1])
             - x[0][1] * (x[1][0] * x[2][2] - x[1][2] * x[2][0])
             + x[0][2] * (x[1][0] * x[2][1] - x[1][1] * x[2][0]);
    }else{
        var sum = 0;
        for (var i = 0; i < x[0].length; i++){
            var xCpy = clone(x);
            var temp = [];
            for (var j = 1; j < x.length; j++){
                temp.push(xCpy[j].slice(0));
                temp[j-1].splice(i, 1);
            }
            if((i % 2) === 0){
                sum = sum + (x[0][i] * determinant(temp));
            }else{
                sum = sum - (x[0][i] * determinant(temp));
            }
        }
        return sum;
    }
}

var inverse = (x) => {
    var minors = (x) => {
        var out = [];
        if (x.length === 2){
            return [[x[1][1],x[1][0]],
                    [x[0][1],x[0][0]]];
        }
        for (var i = 0; i < x.length; i++){
            var temp = [];
            for (var j = 0; j < x.length; j++){
                var minorTemp = clone(x);
                minorTemp.splice(i, 1);
                for (var k = 0; k < minorTemp.length; k++){
                    minorTemp[k].splice(j, 1);
                }
                temp.push(determinant(minorTemp));
            }
            out.push(temp);
        }
        return out;
    }
    var cofactors = (x) => {
        for (var i = 0; i < x.length; i++){
            for (var j = 0; j < x[i].length; j++){
                if (((i % 2===0) && (j % 2 !== 0)) || ((i % 2 !== 0) && (j % 2 === 0))){
                    x[i][j] = (x[i][j] * -1);
                }else{
                    x[i][j] = x[i][j];
                }
            }
        }
        return x;
    }
    var d = determinant(x);
    x = transpose(cofactors(minors(x)));


    for (var i = 0; i < x.length; i++){
        for (var j = 0; j < x[i].length; j++){
            if (((i % 2===0) && (j % 2 !== 0)) || ((i % 2 !== 0) && (j % 2 === 0))){
                x[i][j] = (x[i][j])/d;
            }else{
                x[i][j] = x[i][j]/d;
            }
        }
    }

    return x;
}

var dot = (x, y) => {
    var out = [];
    for(var i = 0; i < x.length; i++){
        var temp = [];
        for(var j = 0; j < y[0].length; j++){
            var sum = 0;
            for(var k = 0; k < y.length; k++){
                sum += x[i][k] * y[k][j];
            }
            temp.push(sum);
        }
        out.push(temp);
    }
    return out;
}

var reLU = (x) => {
    if (x > 0) {
        return x;
    }else{
        return 0;
    }
}

var sigmoid = (x) => {
    return (1 / (1 + Math.pow(Math.E, -x)));
}

var sigmoidPrime = (x) => {
    var eNegX = Math.pow(Math.E, -x);
    return (eNegX / Math.pow(1 + eNegX, 2));
}

var tanh = (x) => {
    var eX = Math.pow(Math.E, x);
    var eNegX = Math.pow(Math.E, -x);
    var out = (eX - eNegX) / (eX + eNegX);
    if (isNaN(out)){
        return 1;
    }
    return out;
}

var tanhPrime = (x) => {
    var tanhx = tanh(x);
    return 1 - (tanhx * tanhx);
}

var linear = (x) => {
    return x;
}

var linearPrime = (x) => {
    return 1;
}

var distance = (x, y) => {
    var sum = 0;
    for (var i = 0; i < x.length; i++){
        sum += Math.pow(x[i] - y[i], 2);
    }
    return Math.sqrt(sum);
}

module.exports = {dot, clone, determinant, inverse, transpose, reLU, sigmoid,
                  sigmoidPrime, tanh, tanhPrime, linear, linearPrime,
                  distance
                 };
