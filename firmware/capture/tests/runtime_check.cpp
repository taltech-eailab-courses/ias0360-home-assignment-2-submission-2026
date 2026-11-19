#include "capture_runtime.h"
#include <cassert>
#include <cstdio>
int main() {
 using namespace capture;
 assert(!autosave_due(19,false,100,2000100));
 assert(!autosave_due(20,false,100,2000099));
 assert(autosave_due(20,false,100,2000100));
 assert(!autosave_due(20,true,100,9999999));
 assert(!autosave_due(20,false,2000100,2000100));
 assert(!autosave_due(100,false,100,99));
 Progress p;
 assert(p.next()==0 && !p.pending() && !p.done());
 assert(p.reserve(0) && p.next()==1 && p.saved==0 && p.pending());
 assert(!p.reserve(1)); // Never reuse a snapshot while the writer owns it.
 assert(!p.finish(1,true)); // A stale/wrong result cannot commit another label.
 assert(p.finish(0,false) && p.failed && p.pending() && p.next()==1);
 assert(p.saved==0 && !p.complete[0]); // A failed save remains retryable.
 assert(p.finish(0,true) && p.saved==1 && !p.pending() && !p.failed);
 assert(!p.finish(0,true) && p.saved==1); // No double counting.
 for(unsigned i=1;i<kTotal;++i) {
  assert(p.next()==i && p.reserve(i));
  if(i==kTotal-1) {
   assert(!p.done() && p.next()==kTotal && p.saved==kTotal-1);
   assert(p.finish(i,false) && !p.done() && p.pending());
  }
  assert(p.finish(i,true));
 }
 assert(p.done() && p.saved==300 && p.next()==300);
 assert(!p.reserve(kTotal));
 // A reboot with an unfinished write requests the first image not on disk.
 Progress reboot;
 reboot.complete[0]=true;reboot.complete[2]=true;reboot.saved=2;
 assert(reboot.next()==1 && reboot.reserve(1) && reboot.next()==3);
 puts("PASS: 20-pixel threshold; 2s idle; no save while touching; snapshot reservation; failure/retry; durable completion; reboot gaps.");
}
