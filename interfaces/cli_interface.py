# cli_interface.py
# 명령줄 인터페이스

import argparse
import sys
import os
from typing import Optional
import json
import time

from core.config import Config, setup_environment
from core.model_manager import get_model_manager
from core.text_generator import TextGenerator, ConversationManager
from core.vision_generator import VisionGenerator, DocumentProcessor

class CLIInterface:
    """명령줄 인터페이스 클래스"""
    
    def __init__(self):
        self.config = None
        self.model_manager = None
        self.text_generator = None
        self.vision_generator = None
        self.conversation_manager = None
        
    def setup(self, config_file: Optional[str] = None):
        """CLI 설정"""
        print("🌸 Korean Bllossom AICA-5B CLI 시작")
        print("=" * 50)
        
        # 환경 설정
        setup_environment()
        
        # 설정 로드
        self.config = Config(config_file)
        self.config.print_config()
        
        # 모델 매니저 초기화
        self.model_manager = get_model_manager(self.config)
        
        # 생성기 초기화
        self.text_generator = TextGenerator(self.model_manager, self.config)
        self.vision_generator = VisionGenerator(self.model_manager, self.config)
        self.conversation_manager = ConversationManager(self.text_generator)
    
    def load_model(self) -> bool:
        """모델 로드"""
        print("\n🚀 모델 로딩 중...")
        success = self.model_manager.load_model()
        
        if success:
            print("✅ 모델 로드 완료!")
            self.model_manager.optimize_for_inference()
            return True
        else:
            print("❌ 모델 로드 실패!")
            return False
    
    def text_mode(self, args):
        """텍스트 생성 모드"""
        if not self.model_manager.is_loaded:
            if not self.load_model():
                return
        
        if args.interactive:
            self._interactive_text_mode(args)
        else:
            self._single_text_generation(args)
    
    def _single_text_generation(self, args):
        """단일 텍스트 생성"""
        prompt = args.prompt or input("프롬프트를 입력하세요: ")
        
        result = self.text_generator.generate(
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            system_message=args.system_message
        )
        
        if result["success"]:
            print(f"\n✨ 생성된 텍스트:\n{result['response']}")
            
            if args.verbose:
                print(f"\n📊 통계:")
                print(f"   생성 시간: {result['generation_time']:.2f}초")
                print(f"   입력 토큰: {result['input_tokens']}")
                print(f"   출력 토큰: {result['output_tokens']}")
                print(f"   속도: {result['tokens_per_second']:.1f} 토큰/초")
            
            if args.output:
                self._save_result(result, args.output)
        else:
            print(f"❌ 생성 실패: {result['error']}")
    
    def _interactive_text_mode(self, args):
        """대화형 텍스트 모드"""
        print("\n💬 대화형 모드 시작!")
        print("명령어:")
        print("  /help - 도움말")
        print("  /clear - 대화 히스토리 초기화")
        print("  /save - 대화 저장")
        print("  /stats - 모델 정보")
        print("  /quit - 종료")
        print("-" * 30)
        
        if args.system_message:
            self.conversation_manager.set_system_message(args.system_message)
            print(f"🤖 시스템 메시지 설정: {args.system_message}")
        
        while True:
            try:
                user_input = input("\n사용자: ").strip()
                
                if not user_input:
                    continue
                
                # 명령어 처리
                if user_input.startswith('/'):
                    if user_input == '/quit':
                        print("👋 대화를 종료합니다.")
                        break
                    elif user_input == '/help':
                        self._print_help()
                    elif user_input == '/clear':
                        self.conversation_manager.clear_history()
                    elif user_input == '/save':
                        filename = self.conversation_manager.export_conversation()
                        if filename:
                            print(f"💾 대화가 저장되었습니다: {filename}")
                    elif user_input == '/stats':
                        self._print_model_stats()
                    else:
                        print("❓ 알 수 없는 명령어입니다. /help를 참조하세요.")
                    continue
                
                # 텍스트 생성
                result = self.conversation_manager.generate_response(
                    user_input,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k
                )
                
                if result["success"]:
                    print(f"AI: {result['response']}")
                    
                    if args.verbose:
                        print(f"[⏱️ {result['generation_time']:.1f}s, 🚀 {result['tokens_per_second']:.1f} t/s]")
                else:
                    print(f"❌ 오류: {result['error']}")
                
                # 메모리 정리
                self.model_manager.clear_memory()
                
            except KeyboardInterrupt:
                print("\n👋 대화를 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
    
    def vision_mode(self, args):
        """시각-언어 모드"""
        if not self.model_manager.is_loaded:
            if not self.load_model():
                return
        
        if args.task == "describe":
            result = self.vision_generator.describe_image(args.image)
        elif args.task == "ocr":
            result = self.vision_generator.extract_text(args.image)
        elif args.task == "chart":
            result = self.vision_generator.analyze_chart(args.image)
        elif args.task == "table":
            result = self.vision_generator.analyze_table(args.image)
        elif args.task == "markdown":
            result = self.vision_generator.convert_to_markdown(args.image)
        elif args.task == "qa":
            if not args.prompt:
                prompt = input("이미지에 대한 질문을 입력하세요: ")
            else:
                prompt = args.prompt
            result = self.vision_generator.answer_visual_question(args.image, prompt)
        else:
            # 사용자 정의 프롬프트
            prompt = args.prompt or "이 이미지에 대해 설명해주세요."
            result = self.vision_generator.generate_with_image(
                args.image,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
        
        if result["success"]:
            print(f"\n✨ 분석 결과:\n{result['response']}")
            
            if args.verbose:
                print(f"\n📊 통계:")
                print(f"   이미지 크기: {result.get('image_size', 'Unknown')}")
                print(f"   생성 시간: {result['generation_time']:.2f}초")
                print(f"   속도: {result['tokens_per_second']:.1f} 토큰/초")
            
            if args.output:
                self._save_result(result, args.output)
        else:
            print(f"❌ 분석 실패: {result['error']}")
    
    def document_mode(self, args):
        """문서 처리 모드"""
        if not self.model_manager.is_loaded:
            if not self.load_model():
                return
        
        doc_processor = DocumentProcessor(self.vision_generator)
        
        if args.multi_page:
            # 다중 페이지 처리
            image_files = []
            if os.path.isdir(args.image):
                # 디렉토리에서 이미지 파일들 찾기
                for file in sorted(os.listdir(args.image)):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        image_files.append(os.path.join(args.image, file))
            else:
                # 단일 파일
                image_files = [args.image]
            
            print(f"📄 {len(image_files)}개 페이지 처리 중...")
            result = doc_processor.process_multi_page_document(image_files, args.task)
            
            if args.output:
                self._save_document_result(result, args.output)
            else:
                print(f"\n📋 문서 처리 결과:\n{result['combined_content']}")
        else:
            # 단일 페이지 처리
            result = doc_processor.process_document_page(args.image, args.task)
            
            if result["success"]:
                print(f"\n📋 문서 처리 결과:\n{result['response']}")
                
                if args.output:
                    self._save_result(result, args.output)
            else:
                print(f"❌ 문서 처리 실패: {result['error']}")
    
    def batch_mode(self, args):
        """배치 처리 모드"""
        if not self.model_manager.is_loaded:
            if not self.load_model():
                return
        
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                if args.input_file.endswith('.json'):
                    batch_data = json.load(f)
                else:
                    # 텍스트 파일 (한 줄에 하나씩)
                    prompts = [line.strip() for line in f if line.strip()]
                    batch_data = [{"prompt": p} for p in prompts]
            
            print(f"📦 배치 처리 시작: {len(batch_data)}개 항목")
            
            results = []
            for i, item in enumerate(batch_data):
                print(f"🔄 처리 중: {i+1}/{len(batch_data)}")
                
                if "image" in item:
                    # 이미지 + 텍스트
                    result = self.vision_generator.generate_with_image(
                        item["image"],
                        item["prompt"],
                        max_tokens=args.max_tokens,
                        temperature=args.temperature
                    )
                else:
                    # 텍스트만
                    result = self.text_generator.generate(
                        item["prompt"],
                        max_tokens=args.max_tokens,
                        temperature=args.temperature
                    )
                
                results.append({
                    "input": item,
                    "output": result,
                    "index": i
                })
                
                # 진행률 출력
                if (i + 1) % 10 == 0:
                    print(f"📊 진행률: {i+1}/{len(batch_data)} ({((i+1)/len(batch_data)*100):.1f}%)")
            
            # 결과 저장
            output_file = args.output or f"batch_results_{int(time.time())}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 배치 처리 완료! 결과 저장: {output_file}")
            
        except Exception as e:
            print(f"❌ 배치 처리 실패: {e}")
    
    def _save_result(self, result: dict, filename: str):
        """결과 저장"""
        try:
            if filename.endswith('.json'):
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(result['response'])
            
            print(f"💾 결과 저장: {filename}")
            
        except Exception as e:
            print(f"❌ 저장 실패: {e}")
    
    def _save_document_result(self, result: dict, filename: str):
        """문서 처리 결과 저장"""
        try:
            if filename.endswith('.json'):
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(result['combined_content'])
            
            print(f"💾 문서 결과 저장: {filename}")
            
        except Exception as e:
            print(f"❌ 저장 실패: {e}")
    
    def _print_help(self):
        """도움말 출력"""
        print("\n📖 도움말:")
        print("  /help - 이 도움말 표시")
        print("  /clear - 대화 히스토리 초기화")
        print("  /save - 현재 대화를 파일로 저장")
        print("  /stats - 모델 및 메모리 정보 표시")
        print("  /quit - 프로그램 종료")
    
    def _print_model_stats(self):
        """모델 통계 출력"""
        info = self.model_manager.get_model_info()
        print(f"\n📊 모델 정보:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        self.model_manager._print_memory_usage()


def create_parser():
    """명령줄 인수 파서 생성"""
    parser = argparse.ArgumentParser(
        description="Korean Bllossom AICA-5B CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예제:
  # 텍스트 생성
  python cli_interface.py text "안녕하세요" --max-tokens 100
  
  # 대화형 모드
  python cli_interface.py text --interactive
  
  # 이미지 설명
  python cli_interface.py vision image.jpg --task describe
  
  # OCR (텍스트 추출)
  python cli_interface.py vision document.png --task ocr
  
  # 문서 처리
  python cli_interface.py document doc.png --task markdown
  
  # 배치 처리
  python cli_interface.py batch prompts.txt --output results.json
        """
    )
    
    # 전역 옵션
    parser.add_argument('--config', type=str, help='설정 파일 경로')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 정보 출력')
    parser.add_argument('--output', '-o', type=str, help='출력 파일 경로')
    
    # 서브 커맨드
    subparsers = parser.add_subparsers(dest='mode', help='실행 모드')
    
    # 텍스트 모드
    text_parser = subparsers.add_parser('text', help='텍스트 생성 모드')
    text_parser.add_argument('prompt', nargs='?', help='입력 프롬프트')
    text_parser.add_argument('--interactive', '-i', action='store_true', help='대화형 모드')
    text_parser.add_argument('--max-tokens', type=int, default=256, help='최대 토큰 수')
    text_parser.add_argument('--temperature', type=float, default=0.7, help='생성 온도')
    text_parser.add_argument('--top-p', type=float, default=0.9, help='Top-p 파라미터')
    text_parser.add_argument('--top-k', type=int, default=50, help='Top-k 파라미터')
    text_parser.add_argument('--system-message', type=str, help='시스템 메시지')
    
    # 시각 모드
    vision_parser = subparsers.add_parser('vision', help='시각-언어 모드')
    vision_parser.add_argument('image', help='이미지 파일 경로 또는 URL')
    vision_parser.add_argument('--task', choices=['describe', 'ocr', 'chart', 'table', 'markdown', 'qa', 'custom'], 
                               default='describe', help='분석 작업')
    vision_parser.add_argument('--prompt', type=str, help='사용자 정의 프롬프트')
    vision_parser.add_argument('--max-tokens', type=int, default=300, help='최대 토큰 수')
    vision_parser.add_argument('--temperature', type=float, default=0.1, help='생성 온도')
    
    # 문서 모드
    doc_parser = subparsers.add_parser('document', help='문서 처리 모드')
    doc_parser.add_argument('image', help='문서 이미지 파일 또는 디렉토리')
    doc_parser.add_argument('--task', choices=['extract', 'summarize', 'markdown', 'table'], 
                            default='extract', help='문서 처리 작업')
    doc_parser.add_argument('--multi-page', action='store_true', help='다중 페이지 문서')
    
    # 배치 모드
    batch_parser = subparsers.add_parser('batch', help='배치 처리 모드')
    batch_parser.add_argument('input_file', help='입력 파일 (JSON 또는 텍스트)')
    batch_parser.add_argument('--max-tokens', type=int, default=256, help='최대 토큰 수')
    batch_parser.add_argument('--temperature', type=float, default=0.7, help='생성 온도')
    
    return parser


def main():
    """메인 함수"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    # CLI 인터페이스 초기화
    cli = CLIInterface()
    cli.setup(args.config)
    
    try:
        # 모드별 실행
        if args.mode == 'text':
            cli.text_mode(args)
        elif args.mode == 'vision':
            cli.vision_mode(args)
        elif args.mode == 'document':
            cli.document_mode(args)
        elif args.mode == 'batch':
            cli.batch_mode(args)
        
    except KeyboardInterrupt:
        print("\n\n👋 프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    finally:
        # 정리
        if cli.model_manager and cli.model_manager.is_loaded:
            cli.model_manager.unload_model()


if __name__ == "__main__":
    main()
