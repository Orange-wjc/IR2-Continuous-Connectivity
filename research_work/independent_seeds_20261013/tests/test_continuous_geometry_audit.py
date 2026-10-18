import unittest
from fractions import Fraction
import numpy as np
from skimage.draw import line
from finite_contact_multimap_report import audit_segments

def trace(a,b):return [{'positions':[a,[0.,0.]]},{'positions':[b,[0.,0.]]}]

def oracle(truth,a,b):
 a=list(map(Fraction,a));b=list(map(Fraction,b));d=[y-x for x,y in zip(a,b)]
 for y,x in zip(*np.nonzero(truth!=255)):
  enter,leave=Fraction(0),Fraction(1);valid=True
  for axis,center in enumerate((int(x),int(y))):
   low,high=Fraction(2*center-1,2),Fraction(2*center+1,2)
   if not d[axis]:
    if not low<a[axis]<high:valid=False;break
   else:
    p,q=(low-a[axis])/d[axis],(high-a[axis])/d[axis]
    enter=max(enter,min(p,q));leave=min(leave,max(p,q))
  if valid and enter<leave:return True
 return False

class GeometryChecks(unittest.TestCase):
 def test_real_rounded_terminal_false_positive_and_subsegment_invariance(self):
  truth=np.full((256,256),255,np.uint8);truth[197,187]=1
  a=[177.,216.];b=[192.76689923806904,183.63636472185829];target=[196.,177.]
  x,y=np.rint(a).astype(int);u,v=np.rint(b).astype(int);yy,xx=line(y,x,v,u)
  self.assertTrue(np.any(truth[yy,xx]!=255))
  with self.assertRaises(AssertionError):audit_segments(truth,trace(a,np.rint(b).tolist()))
  for start,end in [(a,b),(b,a),(a,target),(target,a)]:
   result=audit_segments(truth,trace(start,end),return_details=True)
   self.assertEqual(result['wall_crossings'],0)
   self.assertGreaterEqual(result['boundary_contacts'],1)
 def test_real_penetration_and_inside_endpoint_are_rejected(self):
  truth=np.full((256,256),255,np.uint8);truth[197,187]=1
  for a,b in [([177.,216.],[187.,197.]),([187.,197.],[187.,197.]),([177.,216.],[193.1,183.6])]:
   with self.assertRaises(AssertionError):audit_segments(truth,trace(a,b))
 def test_diagonal_corner_contact_is_distinct_from_interior_crossing(self):
  truth=np.full((256,256),255,np.uint8);truth[128,128]=1
  self.assertEqual(audit_segments(truth,trace([118.,137.],[137.,118.])),2)
  truth[128,127]=1
  with self.assertRaises(AssertionError):audit_segments(truth,trace([118.,137.],[137.,118.]))
 def test_clear_wait_and_boundary_parallel_contact(self):
  truth=np.full((8,8),255,np.uint8);truth[3,3]=1
  self.assertEqual(audit_segments(truth,trace([1.,1.],[1.,1.])),2)
  self.assertEqual(audit_segments(truth,trace([2.5,2.],[2.5,4.])),2)
 def test_bounds_and_nonfinite_positions(self):
  truth=np.full((8,8),255,np.uint8)
  for point in [[-0.1,1.],[8.,1.],[np.nan,1.],[np.inf,1.]]:
   with self.assertRaises(AssertionError):audit_segments(truth,trace([1.,1.],point))
 def test_thousand_random_cases_match_independent_exact_fraction_oracle(self):
  rng=np.random.RandomState(71014)
  for _ in range(1000):
   truth=np.where(rng.rand(7,7)<.22,1,255).astype(np.uint8);truth[0,0]=255
   ai=rng.randint(0,25,2);bi=rng.randint(0,25,2)
   a=[Fraction(int(v),4) for v in ai];b=[Fraction(int(v),4) for v in bi]
   expected=oracle(truth,a,b)
   for start,end in [(a,b),(b,a)]:
    try:audit_segments(truth,trace(list(map(float,start)),list(map(float,end))));actual=False
    except AssertionError:actual=True
    self.assertEqual(actual,expected,(a,b))
if __name__=='__main__':unittest.main()
