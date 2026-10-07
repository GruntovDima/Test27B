"""CPU-only command/provenance safety checks; not inference."""
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

import run_test


class LauncherTests(unittest.TestCase):
    def test_explicit_tp4_context_graph_and_no_speculation(self):
        args=run_test.command(Path('/model'),8001)
        for flag,value in [('--tensor-parallel-size','4'),('--max-num-seqs','4'),
                           ('--max-model-len','20480'),('--port','8001')]:
            self.assertEqual(args[args.index(flag)+1],value)
        self.assertIn('FULL_DECODE_ONLY',args[args.index('--compilation-config')+1])
        self.assertNotIn('--enforce-eager',args)
        self.assertNotIn('--speculative-config',args)
        self.assertIn('--no-enable-prefix-caching',args)

    def test_occupied_port_does_not_start_server(self):
        with patch.object(sys,'argv',['run_test.py','--model-path','/model']), \
             patch('run_test.socket.socket') as sock, patch('run_test.subprocess.Popen') as launch:
            sock.return_value.__enter__.return_value.connect_ex.return_value=0
            with self.assertRaisesRegex(ValueError,'Port occupied'):run_test.main()
            launch.assert_not_called()


if __name__=='__main__':unittest.main()
