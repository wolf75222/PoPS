#include <pops/numerics/moments/affine_velocity.hpp>
#include <chrono>
#include <iostream>
template<int D> void probe(){using B=pops::moments::CartesianMomentBasis<D>;pops::Real old[B::size]{},end[B::size]{},out[B::size]{};old[0]=end[0]=1;for(int i=1;i<B::size;++i)old[i]=end[i]=.01;double checksum=0;int iterations=20000;auto start=std::chrono::steady_clock::now();for(int k=0;k<iterations;++k){end[1]=.01+double(k%17)*1e-5;if(!pops::moments::affine_velocity_push_forward<D>(old,end,.75,.125,out))std::abort();checksum+=out[B::size-1];}auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();std::cout<<D<<","<<B::size<<","<<iterations<<","<<double(ns)/iterations<<","<<checksum<<"\n";}
int main(){probe<2>();probe<4>();probe<5>();}
