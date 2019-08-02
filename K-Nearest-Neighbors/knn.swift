import Foundation


public class dataPoint{
    var xLoc: Float
    var yLoc: Float
    var dataClass: String
    
    init(x:Float,y:Float,c:String)
    {
        self.xLoc = x
        self.yLoc = y
        self.dataClass = c
    }
    
    public func calcDistance(x:Float,y:Float) -> Float
    {
        let forX = (pow((self.xLoc - x), 2))
        let forY = (pow((self.yLoc - y), 2))
        var dist = forX + forY
        dist = sqrt(dist)
        return dist
    }
}


public class KNNAlgorithm
{
    var k:Int
    var dataPoints: [dataPoint]
    
    init(k: Int, dp: [dataPoint])
    {
        self.k = k - 1
        self.dataPoints = dp

    }
    
    public func predict(x: Float, y: Float) -> String
    {
        var classesArray = [String]()
        var distArray = [Float]()
        var closests = [String]()
        
        for point in self.dataPoints {
            distArray.append(point.calcDistance(x: x, y: y))
            classesArray.append(point.dataClass)
        }
        
        
        for _ in 0...k {
            let mindist = distArray.min()!
            let minindex = distArray.firstIndex(of: mindist)!
            let minDistClass = classesArray[minindex]
            closests.append(minDistClass)
            distArray.remove(at: minindex)
        }
        var counts: [String: Int] = [:]
        closests.forEach { counts[$0, default: 0] += 1 }
        
        if counts["A"] ?? 0 > counts["B"] ?? 0{
            return "A"
        } else if counts["A"] ?? 0 < counts["B"] ?? 0{
            return "B"
        } else {
            return "N"
        }
        
    }
}



// TEST DATA
var data = [dataPoint]()
for i in 0...10 {
        data.append(dataPoint(x: -0.1 * Float(i), y: -0.1 * Float(10 - i), c: "A"))
}
for t in 0...10 {
    data.append(dataPoint(x: 0.1 * Float(t), y: 0.1 * Float(10 - t), c: "B"))
}

// INIT ALGORITHM
let Knn = KNNAlgorithm(k: 3, dp: data)

// PREDICTION
Knn.predict(x:-0.1,y:-0.1)

