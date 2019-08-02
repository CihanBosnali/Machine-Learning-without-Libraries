import UIKit


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


public class KMeansAlgorithm
{
    var k:Int
    var epoch: Int
    var dataPoints: [dataPoint]
    var meanCenters: [[Float]]
    
    init(k: Int, epoch: Int, dp: [dataPoint])
    {
        self.k = k - 1
        self.epoch = epoch
        self.dataPoints = dp
        self.meanCenters = []
    }
    
    public func initRandomCenters()
    {
        for _ in 0...self.k
        {
            let center = [Float.random(in: -1 ... 1), Float.random(in: -1 ... 1)]
            self.meanCenters.append(center)
        }
    }
    
    public func findNearestCenter(point: dataPoint) -> String{
        var nearestCenter = 0
        var nearestDistance = Float.greatestFiniteMagnitude
        for i in 0...meanCenters.count-1
        {
            let centerDist = point.calcDistance(x: meanCenters[i][0], y: meanCenters[i][1])

            if centerDist < nearestDistance
            {
                nearestCenter = i
                nearestDistance = centerDist
            }
        }
        return String(nearestCenter)
    }
    
    public func pointClustering()
    {
        for point in dataPoints
        {
            let nearestCenter = findNearestCenter(point: point)
            point.dataClass = String(nearestCenter)
        }
    }
    
    public func changeCenters(){
        var newCenters = [[Float]]()
        for i in 0...meanCenters.count-1
        {
            var totalX = Float(0)
            var totalY = Float(0)
            var numPoints = 0
            
            for point in dataPoints
            {
                if point.dataClass == String(i){
                    numPoints += 1
                    totalX += point.xLoc
                    totalY += point.yLoc
                }
            }
            let newCenterX = totalX / Float(numPoints)
            let newCenterY = totalY / Float(numPoints)
            newCenters.append([newCenterX, newCenterY])
        }
        self.meanCenters = newCenters
    }
    
    public func train()
    {
        initRandomCenters()
        
        for _ in 0...self.epoch
        {
            pointClustering()
            changeCenters()
        }
    }
    
    public func predict(x:Float, y:Float) -> String{
        let point2Predict = dataPoint(x:x, y:y, c: "N")
        let nearestCenter = findNearestCenter(point: point2Predict)
        return nearestCenter
    }
}


// TEST DATA
var randomData = [dataPoint]()
for _ in 0...20 {
    let randX = Float.random(in: -1...1)
    let randY = Float.random(in: -1...1)
    randomData.append(dataPoint(x: randX, y: randY , c: "N"))
}

// TRAIN
var KMeans = KMeansAlgorithm(k: 3, epoch: 5, dp: randomData)
KMeans.train()

// PREDICT
print(KMeans.predict(x: 0.3, y: -0.1))
