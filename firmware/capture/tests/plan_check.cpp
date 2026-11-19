#include "capture_plan.h"
#include <cassert>
#include <cstring>
#include <set>
#include <string>
#include <cstdio>
int main() {
    unsigned train[10]={},val[10]={},test[10]={};
    std::set<std::string> names;
    for(unsigned i=0;i<capture::kTotal;++i) {
        auto item=capture::item(i);
        assert(item.digit<10 && item.round>=1 && item.round<=item.rounds);
        if(!strcmp(item.split,"TRAIN")) ++train[item.digit];
        else if(!strcmp(item.split,"VAL")) ++val[item.digit];
        else { assert(!strcmp(item.split,"TEST")); ++test[item.digit]; }
        char filename[64]; capture::path(filename,sizeof filename,"0:",i);
        assert(names.insert(filename).second);
    }
    for(unsigned d=0;d<10;++d) assert(train[d]==20 && val[d]==5 && test[d]==5);
    assert(capture::kFileSize==67215);
    char p[64]; capture::path(p,sizeof p,"0:",199); assert(!strcmp(p,"0:/H2DIGITS/TRAIN/D9_020.PGM"));
    capture::path(p,sizeof p,"0:",200); assert(!strcmp(p,"0:/H2DIGITS/VAL/D0_001.PGM"));
    capture::path(p,sizeof p,"0:",250); assert(!strcmp(p,"0:/H2DIGITS/TEST/D0_001.PGM"));
    puts("PASS: 300 unique labeled images; 20/5/5 per digit; correct split boundaries and PGM size.");
}
