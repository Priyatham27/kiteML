"""
Tests for Phase 4: Developer Experience & CLI Layer.
"""
import sys
import pytest
from unittest.mock import patch
from kiteml.cli.parser import build_parser

class TestCLI:

    def test_parser_creation(self):
        """Test that the parser is built and has the right subcommands."""
        parser = build_parser()
        
        # Access the subparsers action
        subparsers_action = next(
            (action for action in parser._actions if isinstance(action, type(parser._get_positional_actions()[0])) and action.dest == "command"), 
            None
        )
        
        if not subparsers_action:
            subparsers_action = [a for a in parser._actions if a.dest == "command"][0]
            
        assert subparsers_action is not None
        choices = subparsers_action.choices.keys()
        
        expected_commands = [
            "train", "tr", "predict", "p", "serve", "profile", "doctor",
            "monitor", "export", "init", "experiments", "exp",
            "version", "benchmark", "bench", "dashboard", "plugins", "playground"
        ]
        
        for cmd in expected_commands:
            assert cmd in choices


    @patch("kiteml.cli.commands.doctor.run_doctor")
    def test_doctor_command_dispatch(self, mock_run, monkeypatch):
        """Test that `kiteml doctor` routes correctly."""
        parser = build_parser()
        args = parser.parse_args(["doctor"])
        assert args.func == mock_run


    @patch("kiteml.cli.commands.train.run_train")
    def test_train_command_args(self, mock_run):
        """Test train command argument parsing."""
        parser = build_parser()
        args = parser.parse_args([
            "train", "data.csv", "--target", "churn", "--problem-type", "classification"
        ])
        assert args.data == "data.csv"
        assert args.target == "churn"
        assert args.problem_type == "classification"


    @patch("kiteml.cli.commands.export.run_export")
    def test_export_command_args(self, mock_run):
        """Test export command format enforcement."""
        parser = build_parser()
        
        # Valid
        args = parser.parse_args(["export", "model.kiteml", "--format", "docker"])
        assert args.format == "docker"
        
        # Invalid format should raise SystemExit
        with pytest.raises(SystemExit):
            parser.parse_args(["export", "model.kiteml", "--format", "invalid_format"])
