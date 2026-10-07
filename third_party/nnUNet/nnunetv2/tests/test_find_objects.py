import os
import pickle
import pkgutil
import sys
import unittest
from tempfile import TemporaryDirectory
from textwrap import dedent
from unittest.mock import patch

from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name


def _write_file(path: str, content: str):
    with open(path, "w") as f:
        f.write(dedent(content))


class TestFindObjects(unittest.TestCase):
    def test_external_trainer_in_installed_package_is_picklable(self):
        with TemporaryDirectory() as root:
            trainer_dir = os.path.join(root, "external_trainers")
            os.makedirs(trainer_dir)
            _write_file(os.path.join(trainer_dir, "__init__.py"), "")
            _write_file(
                os.path.join(trainer_dir, "trainer.py"),
                """
                from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


                class PicklableTrainer(nnUNetTrainer):
                    pass
                """,
            )

            sys.path.insert(0, root)
            try:
                with patch.dict(
                    os.environ,
                    {"nnUNet_extTrainer": trainer_dir},
                    clear=False,
                ):
                    trainer_class = recursive_find_trainer_class_by_name("PicklableTrainer")
                    restored = pickle.loads(pickle.dumps(trainer_class))
            finally:
                sys.path.remove(root)
                for module_name in tuple(sys.modules):
                    if module_name == "external_trainers" or module_name.startswith("external_trainers."):
                        sys.modules.pop(module_name)

            self.assertEqual(trainer_class.__module__, "external_trainers.trainer")
            self.assertIs(restored, trainer_class)

    def test_external_trainer_lookup_handles_multiple_directories_with_same_package_name(self):
        with TemporaryDirectory() as first_dir, TemporaryDirectory() as second_dir:
            os.makedirs(os.path.join(first_dir, "sharedpkg"))
            os.makedirs(os.path.join(second_dir, "sharedpkg"))

            _write_file(os.path.join(first_dir, "sharedpkg", "__init__.py"), "")
            _write_file(
                os.path.join(first_dir, "sharedpkg", "trainer.py"),
                """
                from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


                class FirstTrainer(nnUNetTrainer):
                    pass
                """,
            )

            _write_file(os.path.join(second_dir, "sharedpkg", "__init__.py"), "")
            _write_file(
                os.path.join(second_dir, "sharedpkg", "trainer.py"),
                """
                from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


                class SecondTrainer(nnUNetTrainer):
                    pass
                """,
            )

            with patch.dict(
                os.environ,
                {"nnUNet_extTrainer": os.pathsep.join((first_dir, second_dir))},
                clear=False,
            ):
                trainer_class = recursive_find_trainer_class_by_name("SecondTrainer")

            self.assertEqual(trainer_class.__name__, "SecondTrainer")

    def test_external_trainer_lookup_surfaces_import_errors(self):
        with TemporaryDirectory() as trainer_dir:
            os.makedirs(os.path.join(trainer_dir, "brokenpkg"))

            _write_file(os.path.join(trainer_dir, "brokenpkg", "__init__.py"), "")
            _write_file(
                os.path.join(trainer_dir, "brokenpkg", "trainer.py"),
                """
                import definitely_missing_dependency
                from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


                class BrokenTrainer(nnUNetTrainer):
                    pass
                """,
            )

            with patch.dict(
                os.environ,
                {"nnUNet_extTrainer": trainer_dir},
                clear=False,
            ):
                with self.assertRaises(ModuleNotFoundError) as exc:
                    recursive_find_trainer_class_by_name("BrokenTrainer")

            self.assertIn("definitely_missing_dependency", str(exc.exception))


class TestRecursiveFindPythonClass(unittest.TestCase):
    def test_skips_phantom_modules_reported_by_iter_modules(self):
        """pkgutil.iter_modules can occasionally report a module that importlib
        cannot resolve under the dotted path implied by current_module (observed
        under uv on Python >= 3.12). The search must skip such phantoms and
        continue, not abort with ModuleNotFoundError."""

        ModuleInfo = pkgutil.ModuleInfo

        with TemporaryDirectory() as folder:
            _write_file(
                os.path.join(folder, "real_module.py"),
                """
                class TargetClass:
                    pass
                """,
            )

            real = ModuleInfo(None, "real_module", False)
            phantom = ModuleInfo(None, "phantom_module", False)

            with patch(
                "nnunetv2.utilities.find_class_by_name.pkgutil.iter_modules",
                return_value=[phantom, real],
            ):
                result = recursive_find_python_class(
                    folder,
                    "TargetClass",
                    current_module=None,
                )

            self.assertIsNotNone(result)
            self.assertEqual(result.__name__, "TargetClass")

    def test_propagates_real_import_errors_from_module_dependencies(self):
        """If the discovered module's *own dependency* is missing, the
        ModuleNotFoundError must still propagate so users see the real cause."""

        with TemporaryDirectory() as folder:
            _write_file(
                os.path.join(folder, "broken_module.py"),
                """
                import definitely_missing_dependency  # noqa: F401
                """,
            )

            with self.assertRaises(ModuleNotFoundError) as exc:
                recursive_find_python_class(
                    folder,
                    "AnyClass",
                    current_module=None,
                )
            self.assertEqual(exc.exception.name, "definitely_missing_dependency")


if __name__ == "__main__":
    unittest.main()
