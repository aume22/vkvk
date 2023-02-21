#include <algorithm>
#include <cmath>
#include <cstdio>
//#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <omp.h>

using namespace std;

#pragma omp declare reduction(vec_double_plus : std::vector<double> : \
                              std::transform(omp_out.begin(), \
					     omp_out.end(), \
					     omp_in.begin(), \
					     omp_out.begin(), \
					     std::plus<double>())) \
  initializer(omp_priv=decltype(omp_orig)(omp_orig.size()))

/// Square function
constexpr inline double sqr(const double& x)
{
  return x*x;
}

/// Computes the correlation given the volume
vector<double> VKVK(const int& T,const int& L,const double& mu,const int& r12)
{
  /// Quark mass
  const double mu2=
    mu*mu;
  
  /// Phase factor at time boundary in units of 2*M_PI
  constexpr double bc0=
	      0.5;
  
  double pt0[T],ptI[L];
  double p2t0[T],p2tI[L];
  double pc0[T],pcI[L];
  for(int iP0=0;iP0<T;iP0++)
    {
      const double p0=
	2.0*M_PI*(iP0+bc0)/T;
      pt0[iP0]=sin(p0);
      p2t0[iP0]=sqr(pt0[iP0]);
      pc0[iP0]=cos(p0);
    }
  
  for(int iPI=0;iPI<L;iPI++)
    {
      const double pI=
	2.0*M_PI*iPI/L;
      ptI[iPI]=sin(pI);
      p2tI[iPI]=sqr(ptI[iPI]);
      pcI[iPI]=cos(pI);
    }
  
  /// Number of colors
  constexpr int Nc=3;
  
  /// Correlator in momentum space
  vector<double> c(T,0.0);
  
  /// Loop over all Q0, P0 combination
  /// These are arranged as i=iQ0*(iQ0+1)/2+iP0
#pragma omp parallel for reduction(vec_double_plus:c)
  for(int64_t i=0;i<T*(T+1)/2;i++)
    {
      const int iQ0=(-1+sqrt(1+8*i))/2;
      const int iP0=i-(int64_t)iQ0*(iQ0+1)/2;
      
      const int pmq0=(T+iP0-iQ0)%T;
      
      double temp=0;
      for(int iP1=0;iP1<=L/2;iP1++)
	for(int iP2=0;iP2<=iP1;iP2++)
	  for(int iP3=0;iP3<=iP2;iP3++)
	    {
	      /// Space parity
	      const int parMult=
		((iP1 and iP1!=L/2)+1)*
		((iP2 and iP2!=L/2)+1)*
		((iP3 and iP3!=L/2)+1);
	      
	      /// Multiplicity of the permutation, given the number of equal
	      /// components (1 is actually never used)
	      constexpr double permMultTable[]=
			  {1,1,3,6};
	      
	      /// Space components multiplicity
	      const int permMult=
		permMultTable
		[(iP1!=iP2)+
		 (iP2!=iP3)+
		 (iP3!=iP1)];
	      
	      /// M(p)
	      const double Mp=
		4-pc0[iP0]-pcI[iP1]-pcI[iP2]-pcI[iP3];
	      
	      /// M(q)
	      const double Mq=
		4-pc0[iQ0]-pcI[iP1]-pcI[iP2]-pcI[iP3];
	      
	      /// Numerator of the loop integrand
	      const double num=mu2+
		pt0[iP0]*pt0[iQ0]+(ptI[iP1]*ptI[iP1]+ptI[iP2]*ptI[iP2]+ptI[iP3]*ptI[iP3])/3-Mp*Mq*r12;
	      
	      /// First factor of denominator
	      const double dmp=
		(mu2+sqr(Mp)+p2t0[iP0]+p2tI[iP1]+p2tI[iP2]+p2tI[iP3]);
	      
	      /// Second factor of the denominator
	      const double dmq=
		(mu2+sqr(Mq)+p2t0[iQ0]+p2tI[iP1]+p2tI[iP2]+p2tI[iP3]);
	      
	      /// Denominator
	      const double den=
		  dmp*dmq;
	      
	      temp+=
		parMult*permMult*num/den;
	    }
	
      c[pmq0]+=temp*(1+(iP0!=iQ0));
    }
  
  /// Take Fourier transform
  vector<double> d(T,0.0);
  for(int iT=0;iT<T;iT++)
    for(int iP0=0;iP0<T;iP0++)
      d[iT]+=cos(2*M_PI*iP0/T*iT)*c[iP0];
  
  /// Normalization
  const double n=
    4.0*Nc/T/T/L/L/L;
  
  /// Add the normalization
  for(int iT=0;iT<T;iT++)
    d[iT]*=n;
  
  return d;
}

/// Compute the correlation function, given the volume and scale
vector<double> VKVK(const int& T,const int& L,const double& mu,const int& r1,const int& r2,const int& scale,const string& tag)
{
  vector<double> c(T);
  
  const string filePath="corr_"+tag;
  
  bool ex=false;
  {
    ifstream i(filePath);
    if(i.good())
      ex=true;
  }
  
  if(ex)
    {
      int jT;
      ifstream corrFile(filePath);
      for(int iT=0;iT<T;iT++)
	corrFile>>jT>>c[iT];
    }
  else
    {
      vector<double> scaledC=
	VKVK(T*scale,L*scale,mu/scale,r1*r2);
      
      for(int t=0;t<T;t++)
	c[t]=scaledC[t*scale]*scale*scale*scale;
      
      ofstream corrFile(filePath);
      corrFile.precision(17);
      for(int iT=0;iT<T;iT++)
	corrFile<<iT<<" "<<c[iT]<<endl;
    }
  
  return c;
}

double interpolate(const vector<vector<double>>& d,const int& iT,const int& degree)
{
  const int scaleMax=d.size();
  
  vector<double> Al(2*degree+1,0.0);
  vector<double> c(degree+1);
  for(int scale=scaleMax;scale>=scaleMax-degree;scale--)
    {
      const double x=
	1.0/scale/scale;
      
      /// Weight
      double w=
	1.0;
      
      for(int f=0;f<=2*degree;f++)
	{
	  Al[f]+=w;
	  if(f<=degree)
	    c[f]+=d[scale-1][iT]*w;
	  w*=x;
	}
    }
  
  vector<double> A((degree+1)*(degree+1));
  for(int i=0;i<=degree;i++)
    for(int j=0;j<=degree;j++)
      A[i*(degree+1)+j]=Al[i+j];
  
  //
  
  for(int i=0;i<degree+1;i++)
    {
      double C=A[i*(degree+1)+i];
      for(int j=i;j<degree+1;j++)
	A[i*(degree+1)+j]/=C;
      c[i]/=C;
      
      for(int k=i+1;k<degree+1;k++)
	{
	  double C=A[k*(degree+1)+i];
	  for(int j=i;j<degree+1;j++)
	    A[k*(degree+1)+j]-=A[i*(degree+1)+j]*C;
	  c[k]-=C*c[i];
	}
    }
  
  vector<double> res(degree+1);
  for(int k=degree;k>=0;k--)
    {
      double S=
	0.0;
      
      for(int i=k+1;i<degree+1;i++)
	S+=A[k*(degree+1)+i]*res[i];
      res[k]=c[k]-S;
    }
  
  return res[0];
}

int main(int narg,char **arg)
{
  if(narg<4)
    {
      cerr<<"Use: "<<arg[0]<<" L amu scaleMax"<<endl;
      exit(0);
    }
  
#pragma omp parallel
#pragma omp master
  cout<<"Using "<<omp_get_num_threads()<<" threads"<<endl;
  
  const int L=atoi(arg[1]);
  const double mu=strtod(arg[2],nullptr);
  const int scaleMax=atoi(arg[3]);
  
  cout<<"L="<<L<<", mu="<<mu<<", scaleMax="<<scaleMax<<endl;
  
  const int T=L*2;
  constexpr int r1=1;
  
  for(const int& r2 : {-1,1})
    {
      const string physTag=
	"r1_"+to_string((r1+1)/2)+"_r2_"+to_string((r2+1)/2)+"_L_"+to_string(L)+"_T_"+to_string(T)+"_mu_"+arg[2];
	
      vector<vector<double>> c(scaleMax);
      int prevTime=0,initTime=time(0);
      for(int scale=0;scale<scaleMax;scale++)
	{
	  const string tag=physTag+"_scale_"+to_string(scale+1);
	  
	  int curTimeEst=0;
	  int totTimeEst=0;
	  if(scale)
	    {
	      curTimeEst=prevTime*pow((scale+1.0)/scale,5);
	      for(int scalep=scale;scalep<scaleMax;scalep++)
		totTimeEst+=prevTime*pow((scalep+1.0)/scale,5);
	    }
	  
	  cout<<"Computing scale "<<scale+1<<", estimated time: "<<curTimeEst<<" s, total estimated time: "<<totTimeEst<<" s ... ";
	  cout.flush();
	  int curTime=-time(0);
	  c[scale]=VKVK(T,L,mu,r1,r2,1+scale,tag);
	  curTime+=time(0);
	  cout<<"needed time: "<<curTime<<" s, total passed time: "<<time(0)-initTime<<" s"<<endl;
	  
	  prevTime=curTime;
	  
	  ofstream a2CorrFile("a2Corr_"+tag);
	  a2CorrFile.precision(16);
	  for(int iT=0;iT<T;iT++)
	    a2CorrFile<<iT<<" "<<c[scale][iT]-c[0][iT]<<endl;
	}
      
      for(int splineOrder=1;splineOrder<scaleMax;splineOrder++)
	{
	  const string tag=physTag+"_extr_spline_"+to_string(splineOrder);
	  const string filePath="corr_"+tag;
	  
	  vector<double> extr(T);
	  for(int iT=0;iT<T;iT++)
	    extr[iT]=interpolate(c,iT,splineOrder);
	  
	  ofstream corrFile(filePath);
	  corrFile.precision(17);
	  for(int iT=0;iT<T;iT++)
	    corrFile<<iT<<" "<<extr[iT]<<endl;
	  
	  ofstream a2CorrFile("a2Corr_"+tag);
	  a2CorrFile.precision(16);
	  for(int iT=0;iT<T;iT++)
	    a2CorrFile<<iT<<" "<<extr[iT]-c[0][iT]<<endl;
	}
    }
  
  return 0;
}
