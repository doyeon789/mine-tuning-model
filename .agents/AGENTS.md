# 에이전트 작업 규칙 (Codex Guidelines)

1.  하나의 기능을 만든후 해당 파일을 git add 합니다.
    1-1. 여기서 '기능'이라고 하며는 하나의 함수에 파일에 하나의 기능을 각각 의미합니다.
    1-2. 예를 들어 urls에서 post기능을 만들었으면 그것은 'urls post 기능', views에 post함수를 작성하였으면 'views post 기능'으로 구분합니다.
    1-3. 하지만 반복적인 기능에 대해서 같은 파일안에 여러 기능이 존재하고 변하는 코드가 길지 않으면 일괄처리 합니다. (예시 : urls.py)

2.  git add 후 commit을 수행합니다. commit을 수행할때는 다음과 같은 규칙을 따릅니다.
    2-1. feat: 새로운 기능에 대한 커밋
    2-2. fix: 버그 수정에 대한 커밋
    2-3. build: 빌드 관련 파일 수정 / 모듈 설치 또는 삭제에 대한 커밋
    2-4. chore: 그 외 자잘한 수정에 대한 커밋
    2-5. ci: ci 관련 설정 수정에 대한 커밋
    2-6. docs: 문서 수정에 대한 커밋
    2-7. style: 코드 스타일 혹은 포맷 등에 관한 커밋
    2-8. refactor: 코드 리팩토링에 대한 커밋
    2-9. test: 테스트 코드 수정에 대한 커밋
    2-10. perf: 성능 개선에 대한 커밋

    2-11. 커밋 메시지는 <타입>: <대상> <행위> 형식으로 작성합니다. - 대상: 파일명 또는 기능명 (영어, 소문자) - 행위: 한국어 동사구로 마무리

    2-12. 타입별 커밋 메시지 형식 및 예시 - feat : <대상> <행위>
    예) feat: community urls 작성
    예) feat: post views 함수 작성
    예) feat: post model 정의

          - fix   : <대상> <구체적인 문제> 수정
                    예) fix: login 토큰 만료 오류 수정
                    예) fix: post 삭제 시 권한 체크 누락 수정
                    ※ 단순히 "버그 수정"으로 끝내지 않고 무엇이 문제였는지 명시합니다.

          - refactor: <대상> 리팩토링
                    예) refactor: user serializer 리팩토링

          - chore : <대상> <행위>
                    예) chore: community 앱 기본 구조 생성

          - docs  : <대상> 문서 작성/수정
                    예) docs: API 명세 문서 추가

          - style : <대상> 스타일 수정
                    예) style: post views 코드 포맷 정리

          - build : <대상> <행위>
                    예) build: django-rest-framework 패키지 추가

          - test  : <대상> 테스트 작성/수정
                    예) test: post 생성 유닛 테스트 작성

          - perf  : <대상> 성능 개선
                    예) perf: post 목록 쿼리 성능 개선

          - ci    : <대상> <행위>
                    예) ci: github actions 배포 설정 추가

3.  만약 PowerShell 백틱이 JS 문자열과 충돌하면 무시합니다.
