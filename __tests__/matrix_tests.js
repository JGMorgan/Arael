jest.unmock('../libs/ml_math.js');

describe("matrix operations", () => {
    var matrix = require('../libs/ml_math.js');
    it("transpose", () => {

        var x = [[2,3,1],
                 [3,1,1],
                 [2,1,4]];
        expect(matrix.transpose(x)).toEqual([[2,3,2],
                                             [3,1,1],
                                             [1,1,4]]);
    });

    it("dot", () => {
        var x = [[2,3,1],
                 [3,1,1],
                 [2,1,4]];

        expect(matrix.dot(x, x)).toEqual([[15,10,9],
                                          [11,11,8],
                                          [15,11,19]]);
    });

    it("dot using a vector", () => {
        var x = [[2,3,1],
                 [3,1,1],
                 [2,1,4]];
        var y = [[4],[3],[5]]

        expect(matrix.dot(x, y)).toEqual([[22],
                                          [20],
                                          [31]]);
    });

    it("inverse a matrix", () => {
        var x = [[2,3,1],
                 [3,1,1],
                 [2,1,4]];
        expect(matrix.inverse(x)).toEqual([[-3/23,11/23,-2/23],
                                           [10/23,-6/23,-1/23],
                                           [-1/23,-4/23,7/23]]);
    });

    it("inverse a 4x4 matrix", () => {
        var x = [[2,3,1,4],
                 [3,1,1,2],
                 [2,1,4,1],
                 [2,2,3,3]];
        expect(matrix.inverse(x)).toEqual([[1/7,3/7,2/7,-4/7],
                                           [19/14,-3/7,17/14,-27/14],
                                           [-3/14,-1/7,1/14,5/14],
                                           [-11/14,1/7,-15/14,23/14]]);
    });

});
