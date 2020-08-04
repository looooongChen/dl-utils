import numpy as np
from scipy.spatial import distance_matrix

'''
Update: 
    - 10/05/2020 new version beta
Author: Long Chen
Support:
    - higher dimensional data
    - evaluation in 'area' mode and 'line' mode
    - input as label map or stacked binary maps
    - matrics: 
        - averagePrecision, aggregatedPricision
        - averageRecall, aggregatedRecall
        - averageF1, aggregatedF1
        - aggregatedJaccard, instanceAveragedJaccard
        - aggregatedDice, instanceAveragedDice
        - SBD (symmetric best Dice)
'''

def map2stack(map):
    map = np.squeeze(map)
    labels = np.unique(map)
    if map.ndim == 2 and len(labels) > 1:
        stack = []
        for l in labels:
            if l == 0:
                continue
            stack.append(map==l)
        return np.array(stack)>0
    else:
        return None

class Evaluator(object):

    def __init__(self, dimension=2, mode='area', boundary_tolerance=3):

        self.dimension = dimension
        self.mode = mode
        self.boundary_tolerance = boundary_tolerance

        # self.thres = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if thres is None else thres
        # self.thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        self.examples = []
        self.total_pd = 0
        self.total_gt = 0
        

    def add_example(self, pred, gt):
        e = Sample(pred, gt, dimension=self.dimension, mode=self.mode, boundary_tolerance=self.boundary_tolerance)
        self.examples.append(e)

        self.total_pd += e.num_pd
        self.total_gt += e.num_gt
        print("example added, total: ", len(self.examples))

    def mAP(self, thres=None, metric='Jaccard'):

        '''
        Reference about mAP:
            https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
        '''
        
        # thres = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if thres is None else thres
        thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9] if thres is None else thres
        # thres = [0.3] if thres is None else thres

        APs, precision, recall = [], [], []
        for t in thres:
            match_pd = 0
            match_gt = 0
            for e in self.examples:
                m_p, m_g = e.match_num(t, metric=metric)
                match_pd += m_p
                match_gt += m_g
            precision.append(match_pd/self.total_pd)
            recall.append(match_gt/self.total_gt)
            APs.append(match_pd/(self.total_pd+self.total_gt-match_gt))

        for t, p, r, ap in zip(thres, precision, recall, APs):
            print(metric+'_'+str(t)+': ', 'precision ' + str(p) + ', ', 'recall ' + str(r) + ', ', 'AP ' + str(ap) )
        print('meanPrecision', np.mean(precision), 'meanRecall', np.mean(recall), 'mAP', np.mean(APs))
        
        return np.mean(APs)
    
    def aggregatedJaccard(self):
        '''  
        Reference:
            A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology
        '''
        
        agg_intersection, agg_union = 0, 0
        for e in self.examples:
            agg_i, agg_u, _ = e.get_acc_area()
            agg_intersection += agg_i
            agg_union += agg_u
        
        print('aggrated Jaccard: ', agg_intersection/agg_union)

        return agg_intersection/agg_union

    def aggregatedDice(self):

        ''' 
        no defination found, derived from aggrated Jaccard Index
        Reference:
            CNN-BASED PREPROCESSING TO OPTIMIZE WATERSHED-BASED CELL SEGMENTATION IN 3D CONFOCAL MICROSCOPY IMAGES
        '''
        
        agg_intersection, agg_area = 0, 0
        for e in self.examples:
            agg_i, _, agg_a = e.get_acc_area()
            agg_intersection += agg_i
            agg_area += agg_a
        
        print('aggrated Dice: ', 2*agg_intersection/agg_area)

        return 2*agg_intersection/agg_area
        
        


class Sample(object):

    """
    class for evaluating a singe prediction-gt pair
    """

    def __init__(self, pd, gt, dimension=2, mode='area', boundary_tolerance=3):

        '''
        Args:
            pd: numpy array of dimension D or D+1
            gt: numpy array of dimension D or D+1
            dimension: dimension D of the image / ground truth
            mode: 'area' / 'line', evaluate area or boundary
            boundary_tolerance: int, shift tolerance of boundary pixels
        Note:
            pd/gt can be presented by a label map of dimension D, or a binary map of demension (D+1) with each instance occupying one channel of the first dimension. 
            The binary map costs more memory, but can handle overlapped object. If objects are not overlapped, use the label map to save memory and accelarate the computation.
        '''

        assert (pd is not None) and (gt is not None)

        self.dimension = dimension
        self.mode = mode
        self.boundary_tolerance = boundary_tolerance
        self.label_map = (pd.ndim == dimension)
        if self.label_map:
            self.gt, self.pd = gt.astype(np.uint16), pd.astype(np.uint16)
            self.area_gt = {l: c for l, c in zip(*np.unique(self.gt, return_counts=True)) if l != 0}
            self.area_pd = {l: c for l, c in zip(*np.unique(self.pd, return_counts=True)) if l != 0}
        else:
            self.gt, self.pd = gt > 0, pd > 0
            area_gt = np.sum(self.gt, axis=tuple(range(1, 1+dimension)))
            self.area_gt = {l: c for l, c in enumerate(area_gt) if c!=0}
            area_pd = np.sum(self.pd, axis=tuple(range(1, 1+dimension)))
            self.area_pd = {l: c for l, c in enumerate(area_pd) if c!=0}
        
        self.label_gt = list(self.area_gt.keys())
        self.label_pd = list(self.area_pd.keys())
        self.num_gt = len(self.label_gt)
        self.num_pd = len(self.label_pd)

        # the max-overlap match is not symmetric, thus, store them separately
        self.match_pd = None  # (prediction label)-(matched gt label)
        self.intersection_pd = None # (prediction label)-(intersection area)
        self.match_gt = None
        self.intersection_gt = None # (prediction label)-(intersection area)

        # precision 
        self.precision_pd = None
        self.precision_gt = None
        # recall
        self.recall_pd = None
        self.recall_gt = None
        # F1 score
        self.f1_pd = None
        self.f1_gt = None
        # dice
        self.dice_pd = None
        self.dice_gt = None
        # jaccard
        self.jaccard_pd = None
        self.jaccard_gt = None

        # aggreated area
        self.agg_intersection = None
        self.agg_union = None
        self.agg_area = None
    

    def _computePrecision(self, subject='pred'):

        if subject == 'pred' and self.precision_pd is None:    
            self._computeMatch('pred')
            self.precision_pd = {k: self.intersection_pd[k] / self.area_pd[k] for k in self.match_pd.keys()}

        if subject == 'gt' and self.precision_gt is None:    
            self._computeMatch('gt')
            self.precision_gt = {k: self.intersection_gt[k] / self.area_gt[k] for k in self.match_gt.keys()}
        

    def averagePrecision(self, subject='pred'):

        self._computePrecision(subject)
        if subject == 'pred':
            return np.mean(list(self.precision_pd.values()))
        else:
            return np.mean(list(self.precision_gt.values()))


    def aggregatedPrecision(self, subject='pred'):

        self._computeMatch(subject)

        if subject == 'pred':
            intersect_agg = sum(list(self.intersection_pd.values()))
            area_agg = sum(list(self.area_pd.values())) + 1e-8
        else:
            intersect_agg = sum(list(self.intersection_gt.values()))
            area_agg = sum(list(self.area_gt.values())) + 1e-8
        
        return intersect_agg/area_agg
    

    def _computeRecall(self, subject='pred'):

        if subject == 'pred' and self.recall_pd is None:    
            self._computeMatch('pred')
            self.recall_pd = {}
            for k, m in self.match_pd.items():
                self.recall_pd[k] = self.intersection_pd[k] / self.area_gt[m] if m is not None else 0

        if subject == 'gt' and self.recall_gt is None:    
            self._computeMatch('gt')
            self.recall_gt = {}
            for k, m in self.match_gt.items():
                self.recall_gt[k] = self.intersection_gt[k] / self.area_pd[m] if m is not None else 0


    def averageRecall(self, subject='pred'):
        self._computeRecall(subject)
        if subject == 'pred':
            return np.mean(list(self.recall_pd.values()))
        else:
            return np.mean(list(self.recall_gt.values()))    


    def aggregatedRecall(self, subject='pred'):

        self._computeMatch(subject)

        intersect_agg, area_agg = 0, 1e-8
        if subject == 'pred':
            for k, m in self.match_pd.items():
                intersect_agg += self.intersection_pd[k]
                area_agg = area_agg + self.area_gt[m] if m is not None else area_agg
        else:
            for k, m in self.match_gt.items():
                intersect_agg += self.intersection_gt[k]
                area_agg = area_agg + self.area_pd[m] if m is not None else area_agg
            
        return intersect_agg/area_agg
    

    def _computeF1(self, subject='pred'):

        self._computePrecision(subject)
        self._computeRecall(subject)

        if subject == 'pred' and self.f1_pd is None:
            self.f1_pd = {}
            for k, p in self.precision_pd.items():
                self.f1_pd[k] = 2*(p*self.recall_pd[k])/(p + self.recall_pd[k] + 1e-8)

        if subject == 'gt' and self.f1_gt is None:
            self.f1_gt = {}
            for k, p in self.precision_gt.items():
                self.f1_gt[k] = 2*(p*self.recall_gt[k])/(p + self.recall_gt[k] + 1e-8)


    def averageF1(self, subject='pred'):

        self._computeF1(subject)
        if subject == 'pred':
            return np.mean(list(self.f1_pd.values()))
        else:
            return np.mean(list(self.f1_gt.values()))


    def aggregatedF1(self, subject='pred'):
        p = self.aggregatedPrecision(subject)
        r = self.aggregatedRecall(subject)
        return 2*(p*r)/(p+r)


    def _computeJaccard(self, subject='pred'):
        
        self._computeMatch(subject)
        
        if subject == 'pred' and self.jaccard_pd is None:
            match, intersection = self.match_pd, self.intersection_pd
            area_sub, area_ref = self.area_pd, self.area_gt
        elif subject == 'gt' and self.jaccard_gt is None:
            match, intersection = self.match_gt, self.intersection_gt
            area_sub, area_ref = self.area_gt, self.area_pd
        else:
            return None

        jaccard = {}

        for k, m in match.items():
            union = area_sub[k] - intersection[k]
            if m is not None:
                union += area_ref[m]
            jaccard[k] = intersection[k] / union
        
        if subject == 'pred':
            self.jaccard_pd = jaccard
        else:
            self.jaccard_gt = jaccard


    def averagedJaccard(self, subject='pred'):
        '''
        Compute the Jaccard Index for each instance and then take average, note that the result with respect to prediction / ground truth are different
        Args:
            subject: 'pred' or 'gt'
        '''
        self._computeJaccard(subject)
        if subject == 'pred':
            return np.mean(list(self.jaccard_pd.values()))
        else:
            return np.mean(list(self.jaccard_gt.values()))
    
    def get_acc_area(self):
        if self.agg_intersection is None or self.agg_area is None or self.agg_union is None:
            self.agg_intersection, self.agg_union, self.agg_area = 0, 0, 0
            self._computeMatch('gt')
            matched_pd = []
            for k, m in self.match_gt.items():
                self.agg_intersection += self.intersection_gt[k]
                self.agg_union += (self.area_gt[k] - self.intersection_gt[k])
                self.agg_area += self.area_gt[k]
                if m is not None:
                    self.agg_union += self.area_pd[m]
                    self.agg_area += self.area_pd[m]
                    matched_pd.append(m)
            # it is possible that a prediction is matched by multiply gt objects
            self.agg_union += np.sum(list(self.area_pd.values()))
            self.agg_area += np.sum(list(self.area_pd.values()))
            for l in np.unique(matched_pd):
                self.agg_union -= self.area_pd[l]
                self.agg_area -= self.area_pd[l]
        return self.agg_intersection, self.agg_union, self.agg_area


    def aggregatedJaccard(self):
        '''  
        Reference:
            A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology
        '''
        agg_intersection, agg_union, _ = self.get_acc_area()

        return agg_intersection/agg_union


    def _computeDice(self, subject='pred'):

        self._computeMatch(subject)
        if subject == 'pred' and self.dice_pd is None:
            match, intersection = self.match_pd, self.intersection_pd
            area_sub, area_ref = self.area_pd, self.area_gt
        elif subject == 'gt' and self.dice_gt is None:
            match, intersection = self.match_gt, self.intersection_gt
            area_sub, area_ref = self.area_gt, self.area_pd
        else:
            return None

        dice = {}

        for k, m in match.items():
            agg_area = area_sub[k] + area_ref[m] if m is not None else area_sub[k]
            dice[k] = 2 * intersection[k] / agg_area
        
        if subject == 'pred':
            self.dice_pd = dice
        else:
            self.dice_gt = dice


    def averagedDice(self, subject='pred'):
        '''
        Compute the Dice Coefficient for each instance and then take average, note that the result with respect to prediction / ground truth are different
        Args:
            subject: 'pred' or 'gt'
        '''
        self._computeDice(subject)
        if subject == 'pred':
            return np.mean(list(self.dice_pd.values()))
        else:
            return np.mean(list(self.dice_gt.values()))
    

    def aggregatedDice(self):
        ''' 
        no defination found, derived from aggrated Jaccard Index
        Reference:
            CNN-BASED PREPROCESSING TO OPTIMIZE WATERSHED-BASED CELL SEGMENTATION IN 3D CONFOCAL MICROSCOPY IMAGES
        '''
        agg_intersection, _, agg_area = self.get_acc_area()

        return 2*agg_intersection/agg_area


    def SBD(self):
        return min(self.averagedDice('pred'), self.averagedDice('gt'))


    def _computeMatch(self, subject):

        '''
        Args:
            subject: 'pred' or 'gt'
        '''

        if subject == 'pred' and self.match_pd is None:
            sub, ref, label_sub, label_ref = self.pd, self.gt, self.label_pd, self.label_gt
        elif subject == 'gt' and self.match_gt is None:
            sub, ref, label_sub, label_ref = self.gt, self.pd, self.label_gt, self.label_pd
        else:
            return None

        match = {}
        intersection = {}

        for label_s in label_sub:

            if self.mode == "area":
                if self.label_map:
                    overlap = ref[np.nonzero(sub == label_s)]
                    overlap = overlap[np.nonzero(overlap)]
                    if len(overlap) == 0:
                        match[label_s] = None
                        intersection[label_s] = 0
                    else:
                        values, counts = np.unique(overlap, return_counts=True)
                        ind = np.argmax(counts)
                        match[label_s] = values[ind]
                        intersection[label_s] = counts[ind]
                else:
                    overlap = np.sum(np.multiply(ref, np.expand_dims(sub[label_s], axis=0)), axis=tuple(range(1, 1+self.dimension)))
                    ind = np.argsort(overlap, kind='mergesort')
                    if overlap[ind[-1]] == 0:
                        match[label_s] = None
                        intersection[label_s] = 0
                    else:
                        match[label_s] = ind[-1]
                        intersection[label_s] = overlap[ind[-1]]
            else:
                overlap = []
                if self.label_map:
                    pts_sub = np.transpose(np.array(np.nonzero(sub==label_s)))
                    for label_r in label_ref:
                        pts_ref = np.transpose(np.array(np.nonzero(ref==label_r)))
                        bpGraph = distance_matrix(pts_sub, pts_ref) < self.boundary_tolerance
                        overlap.append(GFG(bpGraph).maxBPM())
                else:
                    pts_sub = np.transpose(np.array(np.nonzero(sub[label_s])))
                    for label_r in label_ref:
                        pts_ref = np.transpose(np.array(np.nonzero(ref[label_r])))
                        bpGraph = distance_matrix(pts_sub, pts_ref) < self.boundary_tolerance
                        overlap.append(GFG(bpGraph).maxBPM())
                
                overlap = np.array(overlap)
                ind = np.argsort(overlap, kind='mergesort')
                if overlap[ind[-1]] == 0:
                    match[label_s] = None
                    intersection[label_s] = 0
                else:
                    match[label_s] = label_ref[int(ind)]
                    intersection[label_s] = overlap[ind[-1]]

        if subject == 'pred':
            self.match_pd = match
            self.intersection_pd = intersection
        else:
            self.match_gt = match
            self.intersection_gt = intersection


    def match_num(self, thres, metric='Jaccard'):
        '''
        Args:
            thres: threshold to determine the a match
            metric: metric used to determine match
        Retrun:
            match_count, gt_count: the number of matches, the number of matched gt objects
        '''
        match_count = 0
        match_gt = []
        if metric.lower() == 'f1':
            self._computeF1()
            score = self.f1_pd 
        elif metric.lower() == 'jaccard':
            self._computeJaccard('pred')
            score = self.jaccard_pd
        elif metric.lower() == 'dice':
            self._computeDice('pred')
            score = self.dice_pd
        for k, s in score.items():
            if s >= thres:
                match_count += 1
                if self.match_pd[k] is not None:
                    match_gt.append(self.match_pd[k])
        return match_count, len(np.unique(match_gt))


class GFG(object):   
    # maximal Bipartite matching. 
    def __init__(self, graph): 
          
        self.graph = graph
        # number of applicants  
        self.ppl = len(graph)
        # number of jobs 
        self.jobs = len(graph[0]) 
  
    # A DFS based recursive function 
    # that returns true if a matching  
    # for vertex u is possible 
    def bpm(self, u, matchR, seen): 
        for v in range(self.jobs): 
            # If applicant u is interested in job v and v is not seen 
            if self.graph[u][v] and seen[v] == False: 
                seen[v] = True 
                '''If job 'v' is not assigned to 
                   an applicant OR previously assigned  
                   applicant for job v (which is matchR[v])  
                   has an alternate job available.  
                   Since v is marked as visited in the  
                   above line, matchR[v]  in the following 
                   recursive call will not get job 'v' again'''
                if matchR[v] == -1 or self.bpm(matchR[v], matchR, seen): 
                    matchR[v] = u 
                    return True
        return False
    
    def maxBPM(self): 
        ''' returns maximum number of matching ''' 
        # applicant number assigned to job i, the value -1 indicates nobody is assigned
        matchR = [-1] * self.jobs   
        # Count of jobs assigned to applicants 
        result = 0 
        for i in range(self.ppl): 
            # Mark all jobs as not seen for next applicant. 
            seen = [False] * self.jobs 
            # Find if the applicant 'u' can get a job 
            if self.bpm(i, matchR, seen): 
                result += 1
        return result 


if __name__ == '__main__':
    gt = np.zeros((10,10))
    gt[:4,:4] = 1
    gt[-4:,-4:] = 2

    pred = gt.copy()
    pred[:4,:2] = 3
    # pred[-4:,:2] = 3

    print(gt)
    print(pred)

    # gt = map2stack(gt)
    # pred = map2stack(pred)

    s = Sample(pred, gt, dimension=2, mode='area', boundary_tolerance=3)

    # gt = np.zeros((10,10))
    # gt[2,:] = 1

    # pred = np.zeros((10,10))
    # pred[5,:5] = 1

    # print(gt)
    # print(pred)

    # s = Sample(pred, gt, dimension=2, mode='line', boundary_tolerance=4)

    print('averagePrecision_pd', s.averagePrecision())
    print('averagePrecision_gt', s.averagePrecision('gt'))
    print('aggregatedPrecision_pd', s.aggregatedPrecision())
    print('aggregatedPrecision_gt', s.aggregatedPrecision('gt'))
    print('averageRecall_pd', s.averageRecall())
    print('averageRecall_gt', s.averageRecall('gt'))
    print('aggregatedRecall_pd', s.aggregatedRecall())
    print('aggregatedRecall_gt', s.aggregatedRecall('gt'))
    print('averageF1_pd', s.averageF1())
    print('averageF1_gt', s.averageF1('gt'))
    print('aggregatedF1_pd', s.aggregatedF1())
    print('aggregatedF1_gt', s.aggregatedF1('gt'))

    print('aggregatedJaccard: ', s.aggregatedJaccard())
    print('aggregatedDice: ', s.aggregatedDice())
    print('averageJaccard_pred: ', s.averagedJaccard('pred'))
    print('averageJaccard_gt: ', s.averagedJaccard('gt'))
    print('averageDice_pred: ', s.averagedDice('pred'))
    print('averageDice_gt: ', s.averagedDice('gt'))
    print('SBD: ', s.SBD())
    print('match number', s.match_num(0.1, 'Jaccard'))

    e = Evaluator(dimension=2, mode='area')
    e.add_example(pred, gt)
    e.mAP()
# e.aggregatedJaccard()
# e.aggregatedDice()