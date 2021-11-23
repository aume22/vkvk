#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

#pragma omp declare reduction(vec_double_plus : std::vector<double> : \
                              std::transform(omp_out.begin(), \
					     omp_out.end(), \
					     omp_in.begin(), \
					     omp_out.begin(), \
					     std::plus<double>())) \
  initializer(omp_priv=decltype(omp_orig)(omp_orig.size()))

constexpr inline double sqr(const double& x)
{
  return x*x;
}

vector<double> VKVK(const int& T,const int& L,const double& mu,const int& r12)
{
  const double mu2=mu*mu;
  constexpr double bc0=0.5;
  
  vector<double> c(T,0.0);
  
  // double p0[T],pI[L];
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
  
  constexpr double permMultTable[]=
	      {1,1,3,6};
  
  constexpr int Nc=3;
  
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
	      const int parMult=
		((iP1 and iP1!=L/2)+1)*
		((iP2 and iP2!=L/2)+1)*
		((iP3 and iP3!=L/2)+1);
	      
	      const int permMult=
		permMultTable
		[(iP1!=iP2)+
		 (iP2!=iP3)+
		 (iP3!=iP1)];
	      
	      const double Mp=4-pc0[iP0]-pcI[iP1]-pcI[iP2]-pcI[iP3];
	      const double Mq=4-pc0[iQ0]-pcI[iP1]-pcI[iP2]-pcI[iP3];
	      
	      // const double num=mu2+
	      //   pt0[iP0]*pt0[iQ0]+ptI[iP1]*ptI[iP1]+ptI[iP2]*ptI[iP2]+ptI[iP3]*ptI[iP3]+Mp*Mq*r1*r2;
	      const double num=mu2+
		pt0[iP0]*pt0[iQ0]+(ptI[iP1]*ptI[iP1]+ptI[iP2]*ptI[iP2]+ptI[iP3]*ptI[iP3])/3-Mp*Mq*r12;
	      
	      const double dmp=
		(mu2+sqr(Mp)+p2t0[iP0]+p2tI[iP1]+p2tI[iP2]+p2tI[iP3]);
	      
	      const double dmq=
		(mu2+sqr(Mq)+p2t0[iQ0]+p2tI[iP1]+p2tI[iP2]+p2tI[iP3]);
	      
	      const double den=
		  dmp*dmq;
	      
	      temp+=parMult*permMult*num/den;
	    }
	
      c[pmq0]+=temp*(1+(iP0!=iQ0));
    }
  
  vector<double> d(T,0.0);
  for(int iT=0;iT<T;iT++)
    for(int iP0=0;iP0<T;iP0++)
      d[iT]+=cos(2*M_PI*iP0/T*iT)*c[iP0];
  
  const double n=
    4.0*Nc/T/T/L/L/L;
  
  for(int iT=0;iT<T;iT++)
    d[iT]*=n;
  
  return d;
}

vector<double> VKVK(const int& T,const int& L,const double& mu,const int& r1,const int& r2,const int& scale)
{
  vector<double> scaledC=
    VKVK(T*scale,L*scale,mu,r1*r2);
  
  vector<double> c(T);
  
  for(int t=0;t<T;t++)
    c[t]=scaledC[t*scale]*scale*scale*scale;
  
  return c;
}

int main(int narg,char **arg)
{
  if(narg<3)
    {
      cerr<<"Use: "<<arg[0]<<" L scaleMax"<<endl;
      exit(0);
    }
  
  const int L=atoi(arg[1]);
  const int scaleMax=atoi(arg[2]);
  
  cout<<"L="<<L<<" scaleMax="<<scaleMax<<endl;
  
  const int T=L*2;
  constexpr int r1=1;
  constexpr double mu=0;
  
  for(const int& r2 : {-1,1})
    {
      vector<vector<double>> c(scaleMax);
      int prevTime=0,initTime=time(0);
      for(int scale=0;scale<scaleMax;scale++)
	{
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
	  c[scale]=VKVK(T,L,mu,r1,r2,1+scale);
	  curTime+=time(0);
	  cout<<"needed time: "<<curTime<<" s, total passed time: "<<time(0)-initTime<<" s"<<endl;
	  
	  prevTime=curTime;
	  
	  ofstream corrFile("corrFile_r1_"+to_string((r1+1)/2)+"_r2_"+to_string((r2+1)/2)+"_L_"+to_string(L)+"_T_"+to_string(T)+"_scale_"+to_string(scale+1));
	  for(int iT=0;iT<T;iT++)
	    corrFile<<iT<<" "<<c[scale][iT]-c[0][iT]<<endl;
	}
      
      // for(int iT=0;iT<T;iT++)
      // 	{
      // 	  cout<<iT;
      // 	  for(int scale=0;scale<scaleMax;scale++)
      // 	    cout<<" "<<c[scale][iT];
      // 	  cout<<endl;
      // 	}
    }
  
  return 0;
}
