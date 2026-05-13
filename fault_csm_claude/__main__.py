if __package__ in (None, ""):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fault_csm_claude.main import main

main()
