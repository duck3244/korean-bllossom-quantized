# main.py (수정된 버전)
# Korean Bllossom AICA-5B 메인 실행 파일
# Import 경로 문제 해결

import sys
import os
import argparse
import warnings

warnings.filterwarnings('ignore')

from core.config import Config, setup_environment
from core.model_manager import get_model_manager
from core.text_generator import TextGenerator, ConversationManager
from core.vision_generator import VisionGenerator, DocumentProcessor

def demo_mode():
    """데모 모드 - 모든 기능 테스트"""
    print("🌸 Korean Bllossom AICA-5B 데모 모드")
    print("=" * 60)

    try:
        # 환경 설정
        setup_environment()

        # 설정 로드
        config = Config()
        config.print_config()

        # 모델 매니저 초기화
        print("\n🚀 모델 매니저 초기화 중...")
        model_manager = get_model_manager(config)

        if not model_manager.load_model():
            print("❌ 모델 로드 실패! 데모를 종료합니다.")
            return False

        # 텍스트 생성기 초기화
        text_generator = TextGenerator(model_manager, config)
        vision_generator = VisionGenerator(model_manager, config)

        print("\n" + "=" * 60)
        print("🧪 기능 테스트 시작")

        # 1. 한국어 텍스트 생성 테스트
        print("\n📝 테스트 1: 한국어 텍스트 생성")
        print("-" * 30)

        result1 = text_generator.generate(
            "인공지능의 미래에 대해 간단히 설명해주세요.",
            max_tokens=150,
            temperature=0.7
        )

        if result1["success"]:
            print(f"✅ 성공! 응답: {result1['response']}")
            print(f"📊 속도: {result1['tokens_per_second']:.1f} 토큰/초")
        else:
            print(f"❌ 실패: {result1['error']}")

        # 메모리 정리
        model_manager.clear_memory()

        # 2. 영어 텍스트 생성 테스트
        print("\n📝 테스트 2: 영어 텍스트 생성")
        print("-" * 30)

        result2 = text_generator.generate(
            "What are the main benefits of using AI in healthcare?",
            max_tokens=120,
            temperature=0.6
        )

        if result2["success"]:
            print(f"✅ 성공! 응답: {result2['response']}")
            print(f"📊 속도: {result2['tokens_per_second']:.1f} 토큰/초")
        else:
            print(f"❌ 실패: {result2['error']}")

        # 메모리 정리
        model_manager.clear_memory()

        # 3. 대화 모드 테스트
        print("\n💬 테스트 3: 대화 모드")
        print("-" * 30)

        conversation = ConversationManager(text_generator, max_history=4)
        conversation.set_system_message("당신은 친근하고 도움이 되는 AI 어시스턴트입니다.")

        # 첫 번째 대화
        conv_result1 = conversation.generate_response(
            "안녕하세요! 오늘 날씨가 어떤가요?",
            max_tokens=100,
            temperature=0.8
        )

        if conv_result1["success"]:
            print(f"사용자: 안녕하세요! 오늘 날씨가 어떤가요?")
            print(f"AI: {conv_result1['response']}")

        # 두 번째 대화
        conv_result2 = conversation.generate_response(
            "그럼 실내에서 할 수 있는 활동을 추천해주세요.",
            max_tokens=100,
            temperature=0.8
        )

        if conv_result2["success"]:
            print(f"사용자: 그럼 실내에서 할 수 있는 활동을 추천해주세요.")
            print(f"AI: {conv_result2['response']}")

        # 대화 요약
        summary = conversation.get_history_summary()
        print(f"📋 대화 요약: {summary['total_messages']}개 메시지")

        # 메모리 정리
        model_manager.clear_memory()

        # 4. 시각-언어 테스트 (옵션)
        print("\n🖼️ 테스트 4: 시각-언어 모델 (선택적)")
        print("-" * 30)

        try:
            # 간단한 테스트 이미지 (React 로고)
            test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/1200px-React-icon.svg.png"

            vision_result = vision_generator.describe_image(test_image_url)

            if vision_result["success"]:
                print(f"✅ 이미지 분석 성공!")
                print(f"🖼️ 응답: {vision_result['response']}")
                print(f"📊 속도: {vision_result['tokens_per_second']:.1f} 토큰/초")
            else:
                print(f"⚠️ 이미지 분석 실패: {vision_result['error']}")

        except Exception as e:
            print(f"⚠️ 시각-언어 테스트 건너뛰기: {e}")

        # 최종 메모리 정리
        model_manager.clear_memory()

        print("\n" + "=" * 60)
        print("✅ 모든 테스트 완료!")

        # 최종 메모리 상태
        model_manager._print_memory_usage()

        return True

    except Exception as e:
        print(f"\n❌ 데모 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 모델 언로드
        if 'model_manager' in locals():
            model_manager.unload_model()


def simple_chat_mode():
    """간단한 채팅 모드"""
    print("💬 Korean Bllossom 간단 채팅 모드")
    print("=" * 40)
    print("종료하려면 'quit', 'exit', '종료' 입력")
    print("-" * 40)

    try:
        # 환경 설정
        setup_environment()
        config = Config()

        # 모델 초기화
        model_manager = get_model_manager(config)
        if not model_manager.load_model():
            print("❌ 모델 로드 실패!")
            return

        text_generator = TextGenerator(model_manager, config)
        conversation = ConversationManager(text_generator)
        conversation.set_system_message("당신은 한국어와 영어를 잘하는 친근한 AI 어시스턴트입니다.")

        while True:
            try:
                user_input = input("\n사용자: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                    print("👋 채팅을 종료합니다.")
                    break

                result = conversation.generate_response(
                    user_input,
                    max_tokens=200,
                    temperature=0.7
                )

                if result["success"]:
                    print(f"AI: {result['response']}")
                else:
                    print(f"❌ 오류: {result['error']}")

                # 메모리 정리
                model_manager.clear_memory()

            except KeyboardInterrupt:
                print("\n👋 채팅을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")

    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if 'model_manager' in locals():
            model_manager.unload_model()


def interactive_menu():
    """대화형 메뉴"""
    while True:
        print("\n🌸 Korean Bllossom AICA-5B")
        print("=" * 30)
        print("1. 데모 모드 (모든 기능 테스트)")
        print("2. 간단 채팅 모드")
        print("3. CLI 모드")
        print("4. 설정 확인")
        print("5. 시스템 정보")
        print("0. 종료")
        print("-" * 30)

        try:
            choice = input("선택하세요 (0-5): ").strip()

            if choice == '0':
                print("👋 프로그램을 종료합니다.")
                break
            elif choice == '1':
                demo_mode()
            elif choice == '2':
                simple_chat_mode()
            elif choice == '3':
                print("\nCLI 모드를 사용하려면 다음과 같이 실행하세요:")
                print("python cli_interface.py --help")
                print("예: python cli_interface.py text --interactive")
            elif choice == '4':
                show_config()
            elif choice == '5':
                show_system_info()
            else:
                print("❌ 잘못된 선택입니다.")

        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")


def show_config():
    """설정 정보 표시"""
    print("\n⚙️ 현재 설정")
    print("-" * 20)

    try:
        config = Config()
        config.print_config()

        print(f"\n📁 경로 설정:")
        print(f"   캐시 디렉토리: {config.paths.cache_dir or 'Hugging Face 기본 (~/.cache/huggingface/hub)'}")
        print(f"   로그 디렉토리: {config.paths.log_dir}")
        print(f"   출력 디렉토리: {config.paths.output_dir}")
    except Exception as e:
        print(f"❌ 설정 로드 실패: {e}")


def show_system_info():
    """시스템 정보 표시"""
    print("\n💻 시스템 정보")
    print("-" * 20)

    try:
        import torch
        import psutil
        import platform

        print(f"OS: {platform.system()} {platform.release()}")
        print(f"Python: {platform.python_version()}")
        print(f"PyTorch: {torch.__version__}")

        # CPU 정보
        cpu_count = psutil.cpu_count()
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"CPU: {cpu_count}코어")
        print(f"RAM: {ram_gb:.1f}GB")

        # GPU 정보
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"GPU: {gpu_name}")
            print(f"VRAM: {total_vram:.1f}GB")
            print(f"CUDA: {torch.version.cuda}")
        else:
            print("GPU: CUDA 사용 불가")

        # 필수 라이브러리 확인
        print(f"\n📚 라이브러리 확인:")

        packages_to_check = [
            ('transformers', 'transformers'),
            ('accelerate', 'accelerate'),
            ('bitsandbytes', 'bitsandbytes'),
            ('config', 'config'),
            ('model_manager', 'model_manager'),
            ('text_generator', 'text_generator'),
            ('vision_generator', 'vision_generator')
        ]

        for package_name, import_name in packages_to_check:
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'Unknown')
                print(f"   ✅ {package_name}: {version}")
            except ImportError:
                print(f"   ❌ {package_name}: 설치되지 않음")

    except Exception as e:
        print(f"❌ 시스템 정보 수집 실패: {e}")


def check_requirements():
    """필수 요구사항 확인"""
    print("🔍 시스템 요구사항 확인 중...")

    issues = []

    try:
        # Python 버전 확인
        if sys.version_info < (3, 8):
            issues.append("Python 3.8 이상이 필요합니다.")

        # PyTorch 및 CUDA 확인
        try:
            import torch
            if not torch.cuda.is_available():
                issues.append("CUDA가 사용 불가능합니다. GPU 가속을 사용할 수 없습니다.")
            else:
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if total_vram < 6:
                    issues.append(f"VRAM이 {total_vram:.1f}GB로 부족할 수 있습니다. (권장: 8GB 이상)")
        except ImportError:
            issues.append("PyTorch가 설치되지 않았습니다.")

        # 필수 파일 확인
        # required_files = ['config.py', 'model_manager.py', 'text_generator.py', 'vision_generator.py']
        # for file in required_files:
        #     if not os.path.exists(file):
        #         issues.append(f"필수 파일이 없습니다: {file}")

        # 필수 라이브러리 확인
        required_packages = ['transformers', 'accelerate', 'bitsandbytes']
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                issues.append(f"{package} 패키지가 설치되지 않았습니다.")

        # 메모리 확인
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        if ram_gb < 16:
            issues.append(f"RAM이 {ram_gb:.1f}GB로 부족할 수 있습니다. (권장: 32GB 이상)")

        if issues:
            print("\n⚠️ 발견된 문제:")
            for issue in issues:
                print(f"   - {issue}")
            print("\n💡 설치 가이드를 참조하여 문제를 해결하세요.")
            return False
        else:
            print("✅ 모든 요구사항이 충족되었습니다!")
            return True

    except Exception as e:
        print(f"❌ 요구사항 확인 중 오류: {e}")
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Korean Bllossom AICA-5B 프로젝트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실행 모드:
  --demo      : 모든 기능 테스트 (기본값)
  --chat      : 간단한 채팅 모드
  --check     : 시스템 요구사항 확인
  --config    : 설정 정보 표시
  --info      : 시스템 정보 표시

예제:
  python main.py                 # 대화형 메뉴
  python main.py --demo          # 데모 모드
  python main.py --chat          # 채팅 모드
  python main.py --check         # 요구사항 확인
        """
    )

    parser.add_argument('--demo', action='store_true', help='데모 모드 실행')
    parser.add_argument('--chat', action='store_true', help='채팅 모드 실행')
    parser.add_argument('--check', action='store_true', help='시스템 요구사항 확인')
    parser.add_argument('--config', action='store_true', help='설정 정보 표시')
    parser.add_argument('--info', action='store_true', help='시스템 정보 표시')
    parser.add_argument('--config-file', type=str, help='설정 파일 경로')

    args = parser.parse_args()

    try:
        # 환경 설정
        if 'setup_environment' in globals():
            setup_environment()

        # 모드별 실행
        if args.check:
            check_requirements()
        elif args.config:
            show_config()
        elif args.info:
            show_system_info()
        elif args.demo:
            if check_requirements():
                demo_mode()
        elif args.chat:
            if check_requirements():
                simple_chat_mode()
        else:
            # 인수가 없으면 대화형 메뉴 실행
            interactive_menu()

    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()